import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os # 一時ファイルの削除に必要
import queue
import altair as alt # グラフ描画用 
import time
import asyncio
import logging
from pydub import AudioSegment

#機械学習関連のインポート
from ml_model import VoiceQualityModel, generate_training_data

# 音声分析関連のインポート
from voice_analysis import VoiceFeatureExtractor, plot_audio_analysis, evaluate_clarity, get_feedback

# WebRTC関連のライブラリのインポート:FFmpeg警告の無視設定を追加
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# WebRTC関連のインポート
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True# ブラウザで音声を録音するためのライブラリ
except ImportError:
    WEBRTC_AVAILABLE = False
    # ダミー関数を定義（エラーを防ぐため）
    def webrtc_streamer(*args, **kwargs):
        return None
    class WebRtcMode:
        SENDONLY = "sendonly"
    class RTCConfiguration:
        def __init__(self, *args, **kwargs):
            pass


try:
    import av
    import scipy.io.wavfile
except ImportError:
    pass


# セッション状態の初期化
if 'volume_history' not in st.session_state:
    st.session_state.volume_history = []  # 音量履歴
if 'last_sound_time' not in st.session_state:
    st.session_state.last_sound_time = time.time()  # 最後に音が検出された時間
if 'current_drop_rate' not in st.session_state:
    st.session_state.current_drop_rate = 0  # 現在の音量低下率
if 'end_of_sentence_detected' not in st.session_state:
    st.session_state.end_of_sentence_detected = False  # 文末検出フラグ
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []  # フィードバック履歴
if 'page' not in st.session_state:
    st.session_state.page = "ホーム"  # 現在のページ

# 機械学習関連：音声分析モデルの初期化
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = VoiceQualityModel()  # 音声品質モデルのインスタンス
    print("音声品質モデルのインスタンスを作成しました。")

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False  # モデルの訓練状態


# 録音関連
if 'recording' not in st.session_state:
    st.session_state.recording = False  # 録音中かどうかのフラグ
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None  # 録音済み音声データ
if 'temp_audio_file' not in st.session_state:
    st.session_state.temp_audio_file = None  # 一時保存用の音声ファイルパス    
if 'is_capturing' not in st.session_state:
    st.session_state.is_capturing = False  # 音声キャプチャ中かどうかのフラグ
if 'capture_buffer' not in st.session_state:
    st.session_state.capture_buffer = None  # 音声キャプチャ用バッファ
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False  # 音声分析完了フラグ
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None  # 最後の分析結果


# ユーザー導線関連（新規追加）
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
if 'user_guide_completed' not in st.session_state:
    st.session_state.user_guide_completed = False
if 'practice_count' not in st.session_state:
    st.session_state.practice_count = 0
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = False

# リアルタイム機能用
if 'silence_threshold' not in st.session_state:
    st.session_state.silence_threshold = -40
if 'min_silence_duration' not in st.session_state:
    st.session_state.min_silence_duration = 300

# アプリケーション設定
st.set_page_config(
    page_title="語尾までしっかりマスター",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)


# カスタムCSS（アプリのUI）
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .feedback-box {
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .feedback-good {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .feedback-medium {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
    }
    .feedback-bad {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .guide-box {
        background-color: #F3E5F5;
        border: 2px solid #9C27B0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .step-completed {
        color: #4CAF50;
        font-weight: bold;
    }
    .step-current {
        color: #2196F3;
        font-weight: bold;
    }
    .step-pending {
        color: #9E9E9E;
    }
    .next-step {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# 会話サンプル
CONVERSATION_SAMPLES = {
    "家族との会話": [
        "今日の夕食、パスタにするね",
        "さっきのニュース見た？なんか面白かったね",
        "昼間に、加藤さんが来た",
        "土曜日は何か予定ある？1時に集合ね"
    ],
    "友人との会話": [
        "この間の話の続きなんだけど、結局どうなったの？",
        "拍手したら、最後に握手してくれたよ",
        "新しいカフェ見つけたんだ。今度一緒に行かない？",
        "最近どう？何か変わったことあった？"
    ],
    "恋人との会話": [
        "ちょっとこれ手伝ってもらえる？すぐ終わるから",
        "窓開けてもらってもいい？ちょっと暑いと思って",
        "タクシー呼んだけど、坂の上に止まってる",
        "あのね、昨日見た映画がすごく良かったんだ"
    ]
}
def show_progress_indicator():
    """進捗インジケータを表示"""
    col1, col2, col3 = st.columns(3)
    
    # Step 1: アプリのご案内
    with col1:
        if st.session_state.user_guide_completed:
            st.markdown("**1. アプリのご案内**", unsafe_allow_html=True)
        elif st.session_state.page == "ホーム":
            st.markdown("**1. アプリのご案内**", unsafe_allow_html=True)
        else:
            st.markdown("**1. アプリのご案内**", unsafe_allow_html=True)
    
    # Step 2: AI準備
    with col2:
        if st.session_state.model_trained:
            st.markdown("**2. AI準備完了**", unsafe_allow_html=True)
        elif st.session_state.page == "モデル訓練":
            st.markdown("**2. AI準備中**", unsafe_allow_html=True)
        else:
            st.markdown("**2. AI準備**", unsafe_allow_html=True)
    
    # Step 3: 練習開始
    with col3:
        if st.session_state.practice_count > 0:
            st.markdown(f"**3. 練習中 ({st.session_state.practice_count}回)**", unsafe_allow_html=True)
        elif st.session_state.page == "練習を始める":
            st.markdown("**3. 練習開始**", unsafe_allow_html=True)
        else:
            st.markdown("**3. 練習開始**", unsafe_allow_html=True)

def show_next_step_guide():
    """次のステップガイドを表示"""
    if not st.session_state.user_guide_completed and st.session_state.page == "ホーム":
        st.markdown("""
        <div class="next-step">
        <h4>次のステップ</h4>
        <p>まずは <strong>「モデル訓練」</strong> ページでAIを準備しましょう！</p>
        <p>AIを訓練することで、より正確な音声分析が可能になります。</p>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.model_trained and st.session_state.page != "モデル訓練":
        st.markdown("""
        <div class="next-step">
        <h4>AIの準備が必要です</h4>
        <p><strong>「モデル訓練」</strong> ページでAIを準備してから練習を始めることをお勧めします。</p>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.model_trained and st.session_state.practice_count == 0:
        st.markdown("""
        <div class="next-step">
        <h4>AI準備完了！</h4>
        <p><strong>「練習を始める」</strong> ページで音声練習を開始しましょう！</p>
        </div>
        """, unsafe_allow_html=True)

def show_user_guide():
    """初回利用者向けガイドを表示"""
    if st.session_state.first_visit and not st.session_state.user_guide_completed:
        st.markdown("""
        <div class="guide-box">
        <h3>初めての方へ</h3>
        <p>このアプリは3つのステップで構成されています：</p>
        <ol>
            <li><strong>ホーム</strong>: アプリの説明を読む</li>
            <li><strong>モデル訓練</strong>: AIを準備する（1回だけ）</li>
            <li><strong>練習を始める</strong>: 実際に音声練習をする</li>
        </ol>
        <p>本アプリについて、まずは下の説明を読んでください</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("ガイドを読み進める", key="continue_guide"):
                st.session_state.show_guide = True
                st.rerun()
        with col2:
            if st.button("ガイドをスキップ", key="skip_guide"):
                st.session_state.user_guide_completed = True
                st.session_state.first_visit = False
                st.rerun()

# 音声フレームを処理するコールバック関数（リアルタイム用）
def audio_frame_callback(frame):
    """音声フレームを処理するコールバック関数"""
    try:
        sound = frame.to_ndarray()
        
        if len(sound) > 0:
            audio_data = sound.flatten()
            rms = np.sqrt(np.mean(audio_data**2))
            db = 20 * np.log10(max(rms, 1e-10))
            
            st.session_state.volume_history.append({"音量": db})
            if len(st.session_state.volume_history) > 100:
                st.session_state.volume_history.pop(0)
            
            silence_threshold = st.session_state.get('silence_threshold', -40)
            min_silence_duration = st.session_state.get('min_silence_duration', 500)
            
            if db > silence_threshold:
                st.session_state.last_sound_time = time.time()
                st.session_state.end_of_sentence_detected = False
            else:
                current_time = time.time()
                silence_duration = (current_time - st.session_state.last_sound_time) * 1000
                
                if silence_duration > min_silence_duration and not st.session_state.end_of_sentence_detected:
                    st.session_state.end_of_sentence_detected = True
                    
                    if len(st.session_state.volume_history) > 10:
                        recent_volumes = [item["音量"] for item in st.session_state.volume_history[-10:]]
                        
                        if len(recent_volumes) > 5:
                            before_avg = sum(recent_volumes[-7:-4]) / 3
                            after_avg = sum(recent_volumes[-3:]) / 3
                            drop_rate = (before_avg - after_avg) / (abs(before_avg) + 1e-10)
                            
                            st.session_state.current_drop_rate = drop_rate
                            
                            feedback = get_feedback(drop_rate)
                            st.session_state.feedback_history.append({
                                "time": time.strftime("%H:%M:%S"),
                                "drop_rate": drop_rate,
                                "level": feedback["level"],
                                "message": feedback["message"],
                                "emoji": feedback["emoji"]
                            })
    
    except Exception as e:
        logger.error(f"音声フレーム処理エラー: {e}")
    
    return frame

# リアルタイム音量メーターの表示
def display_volume_meter(placeholder):
    if len(st.session_state.volume_history) > 0:
        df = pd.DataFrame(st.session_state.volume_history)
        df = df.reset_index().rename(columns={"index": "時間"})
        

        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("時間", axis=None),
            y=alt.Y("音量", title="音量 (dB)", scale=alt.Scale(domain=[-80, 0]))
        ).properties(
            height=200,
            width='container'
        )
        

        placeholder.altair_chart(chart, use_container_width=True)


# フィードバック履歴の表示
def display_feedback_history(placeholder):
    if len(st.session_state.feedback_history) > 0:
        placeholder.subheader("フィードバック履歴")
        
        for i, feedback in enumerate(reversed(st.session_state.feedback_history[-5:])):
            level = feedback["level"]
            css_class = f"feedback-box feedback-{level}"
            
            placeholder.markdown(
                f"<div class='{css_class}'>"
                f"<p>{feedback['time']} - {feedback['emoji']} {feedback['message']}</p>"
                f"<p>文末の音量低下率: {feedback['drop_rate']:.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

def main():
    # セッション状態の初期化（既にグローバルで実施済み）
    
    # 特徴抽出器の初期化
    feature_extractor = VoiceFeatureExtractor()

    # アプリのタイトル
    st.markdown('<h1 class="main-header">語尾までしっかりマスター</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">身近な会話をしっかり伝えることで、大切な人とのコミュニケーションを高めよう</p>', unsafe_allow_html=True)

    # 進捗インジケータの表示
    show_progress_indicator()
    
    # サイドバーでナビゲーション（改善版）
    st.sidebar.title("メニュー")
    pages = ["ホーム", "練習を始める", "モデル訓練"]
    page = st.sidebar.selectbox("ページを選択", pages, index=pages.index(st.session_state.page))
    st.session_state.page = page

    # ページの状態を表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("現在の状態")
    
    # モデル訓練状況
    if st.session_state.model_trained:
        st.sidebar.success("AI準備完了")
    else:
        st.sidebar.warning("AI未準備")
    
    # 練習回数
    st.sidebar.info(f"🏃‍♂️ 練習回数: {st.session_state.practice_count}回")

    # サイドバーに設定を追加（リアルタイム機能使用時のみ）
    if page == "練習を始める" and WEBRTC_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.title("リアルタイム設定")
        st.session_state.silence_threshold = st.sidebar.slider(
            "無音しきい値 (dB)", 
            -80, 0, st.session_state.silence_threshold,
            help="音声を「無音」と判断する音量レベルを設定します。"
        )
        st.session_state.min_silence_duration = st.sidebar.slider(
            "最小無音時間 (ms)", 
            100, 500, st.session_state.min_silence_duration,
            help="この時間以上の無音が続いた場合に「無音区間」と判断します。"
        )

   # ページごとの表示内容
    if page == "ホーム":
        # 初回利用者ガイド
        show_user_guide()
        
        # メインコンテンツ
        st.markdown('<h2 class="sub-header">🏠 ようこそ！</h2>', unsafe_allow_html=True)
        
        # アプリの説明を詳しく
        st.markdown("""
        <div class="info-box">
        <h3>このアプリの目的</h3>
        <p>このアプリは、日本語の短い会話を分析し、<strong>文末の明瞭さ</strong>を高めることに注力しています。</p>
        <p>日本語は言語の特徴上、自然と語尾の音声量が低下しがちです。</p>
        <p>あなたの発話を分析して、話し方を改善するヒントを提供します。</p>
        </div>
        """, unsafe_allow_html=True)

        # 日本語の特徴について
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
            <h4>🔤 日本語の特徴</h4>
            <ul>
                <li><strong>SOV構造</strong>: 日本語では重要な情報が文末に来る傾向があります</li>
                <li><strong>音量低下</strong>: 話している間に自然と声が小さくなる傾向があります</li>
                <li><strong>親密な会話</strong>: 家族や友人との会話は特に声を落としがちです</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
            <h4>✨ このアプリでできること</h4>
            <ul>
                <li><strong>音声分析</strong>: あなたの話し方を客観的に分析します</li>
                <li><strong>AI評価</strong>: 機械学習によるより詳細な判定が可能です</li>
                <li><strong>改善アドバイス</strong>: よりよいコミュニケーションのためのヒントを届けます</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # 次のステップガイド
        show_next_step_guide()
        
        # 使い方の説明ボタン
        st.markdown("---")
        if st.button("📖 詳しい使い方を見る", key="show_detailed_guide"):
            st.session_state.show_guide = True

        # 使い方ガイドの表示
        if st.session_state.show_guide:
            st.markdown('<h3 class="sub-header">📖 使い方ガイド</h3>', unsafe_allow_html=True)

            with st.expander("STEP 1: AIを準備する（最初に1回だけ）", expanded=True):
                st.write("""
                1. **左メニューから「モデル訓練」ページに移動**
                2. **「モデル訓練を開始」ボタンをクリック**
                3. **AIが学習を完了するまで待つ（約1-2分）**
                4. **特徴量の重要度グラフを確認**
                
                AIを訓練することで、より正確な音声分析が可能になります！
                """)
            
            with st.expander("STEP 2: 音声で練習する"):
                st.write("""
                1. **左メニューから「練習を始める」ページに移動**
                2. **会話カテゴリとサンプル文を選択**
                3. **音声ファイルをアップロードまたは録音**
                4. **分析結果とアドバイスを確認**
                5. **改善点を意識して再度練習**
                
                練習を重ねることで、確実に話し方が改善されます！
                """)
            
            with st.expander("STEP 3: 継続的な改善"):
                st.write("""
                1. **定期的に練習を行う**
                2. **異なる会話サンプルで試す**
                3. **AIとルールベース両方の結果を比較**
                4. **日常会話に意識を取り入れる**
                
                意識を高めると自然な話し方が意識できます。リモート会議などでもプラスの効果
                """)
            
            if st.button("理解しました"):
                st.session_state.user_guide_completed = True
                st.session_state.first_visit = False
                st.success("さっそく左のメニューの「モデル訓練」から始めましょう！")

    elif page == "練習を始める":
        st.markdown('<h2 class="sub-header">音声練習</h2>', unsafe_allow_html=True)
        
        # モデル未訓練時の警告
        if not st.session_state.model_trained:
            st.warning("""
            　**AI未準備の状態です**
            
            より正確な分析のために、先に「モデル訓練」ページでAIを準備することをお勧めします。
            現在はルールベースの分析のみ利用可能です。
            """)
        
        # カテゴリーとサンプル文の選択
        st.markdown('<h3 class="sub-header">📝 練習内容の選択</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("会話カテゴリーを選択", list(CONVERSATION_SAMPLES.keys()))
        with col2:
            sample_index = st.selectbox(
                "サンプル文を選択", 
                range(len(CONVERSATION_SAMPLES[category])),
                format_func=lambda i: CONVERSATION_SAMPLES[category][i]
            )
        
        selected_sample = CONVERSATION_SAMPLES[category][sample_index]
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("### 読み上げるサンプル文")
        st.markdown(f"**「{selected_sample}」**")
        st.write("このサンプル文を、普段のように自然に読み上げてください。")
        st.markdown('</div>', unsafe_allow_html=True)

        # 練習方法の選択
        st.markdown('<h3 class="sub-header">🎙️ 録音方法の選択</h3>', unsafe_allow_html=True)
        
        if WEBRTC_AVAILABLE:
            practice_method = st.radio(
                "練習方法を選択", 
                ["音声ファイルをアップロード", "リアルタイム評価"],
                help="ファイルアップロード: 録音済み音声を分析 / リアルタイム評価: その場で録音して即座に評価"
            )
        else:
            practice_method = st.radio("練習方法を選択", ["音声ファイルをアップロード"])
            st.info("リアルタイム評価を使用するには、`streamlit-webrtc`をインストールしてください。")

        if practice_method == "音声ファイルをアップロード":
            uploaded_file = st.file_uploader(
                "音声ファイルをアップロードしてください", 
                type=["wav", "mp3"],
                key="file_uploader",
                help="WAVまたはMP3形式の音声ファイルを選択してください"
            )

def handle_file_upload(feature_extractor):
    """ファイルアップロード処理（録音ソース検出付き）"""
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロードしてください", 
        type=["wav", "mp3"],
        key="file_uploader",
        help="WAVまたはMP3形式の音声ファイルを選択してください"
    )

    # 録音ソースの選択
    st.markdown("### 録音環境の選択")
    recording_source = st.radio(
        "この音声はどのように録音されましたか？",
        ["file", "microphone", "smartphone"],
        format_func=lambda x: {
            "file": "ファイル（録音環境不明）",
            "microphone": "マイク録音（PC・専用マイク）", 
            "smartphone": "スマートフォン録音"
        }[x],
        help="録音環境により音量補正の方法が変わります"
    )

    # 会話環境の選択
    conversation_context = st.selectbox(
        "会話の状況を選択してください",
        ["casual", "intimate", "very_intimate", "formal"],
        format_func=lambda x: {
            "casual": "普通の会話（友人同士）",
            "intimate": "親密な会話（恋人・親しい友人）",
            "very_intimate": "非常に親密（家族間の小声）",
            "formal": "フォーマル（会議・発表）"
        }[x],
        help="親密さのレベルにより評価基準が調整されます"
    )
    
    if uploaded_file is not None:
        tmp_file_path = None
        try:
            with st.spinner("音声ファイルを処理中..."):
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 音声ファイルを再生可能に表示
                st.audio(tmp_file_path, format='audio/wav')

                # 音声データの読み込み
                y, sr = librosa.load(tmp_file_path, sr=None)

            # 分析開始（録音ソースと会話環境を考慮）
            with st.spinner("音声を分析中..."):
                analyze_audio_with_context(feature_extractor, y, sr, recording_source, conversation_context)

        except Exception as e:
            st.error(f"音声分析中にエラーが発生しました: {e}")
            logger.error(f"音声分析エラー: {e}", exc_info=True)

        finally:
            # 一時ファイルを削除
            if tmp_file_path is not None and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"一時ファイル削除エラー: {cleanup_error}")

def analyze_audio_with_context(feature_extractor, y, sr, recording_source, conversation_context):
    """録音環境と会話コンテキストを考慮した音声分析"""
    
    # 録音ソースを特徴抽出に渡す
    features = feature_extractor.extract_features(y, sr, recording_source)
    
    # 会話コンテキストを特徴に追加
    features['conversation_context'] = conversation_context
    
    # 練習回数をカウント
    st.session_state.practice_count += 1
    
    # コンテキスト情報の表示
    st.markdown('###録音・会話環境')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**録音方法**: {recording_source}")
        
    with col2:
        st.info(f"**会話レベル**: {conversation_context}")
        
    with col3:
        intimacy = features.get('intimacy_level', 'unknown')
        st.info(f"**AI判定親密さ**: {intimacy}")
    
    # 音量補正情報の表示
    if 'volume_adjustment_ratio' in features:
        ratio = features['volume_adjustment_ratio']
        if ratio != 1.0:
            if ratio < 1.0:
                st.success(f"音量を{1/ratio:.1f}倍に調整しました（親密な会話レベルに補正）")
            else:
                st.info(f"音量を{ratio:.1f}倍に調整しました（親密な会話レベルに補正）")
    
    # 音声分析の視覚化
    st.markdown('<h3 class="sub-header">音声分析結果</h3>', unsafe_allow_html=True)
    fig = plot_audio_analysis(features, y, sr)
    st.pyplot(fig)
    
    # 音量分析結果の表示
    display_volume_analysis_with_context(features)
    
    # 総合分析結果
    display_comprehensive_analysis_with_context(features)

def display_volume_analysis_with_context(features):
    """コンテキスト考慮の音量分析表示"""
    st.markdown('<h3 class="sub-header">音量分析詳細</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("補正後平均音量", f"{features['mean_volume']:.4f}")
        if 'original_mean_volume' in features:
            st.metric("元の平均音量", f"{features['original_mean_volume']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("文頭音量", f"{features['start_volume']:.4f}")
        st.metric("文中音量", f"{features['middle_volume']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("文末音量", f"{features['end_volume']:.4f}")
        st.metric("文末低下率", f"{features['end_drop_rate']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("最後20%音量", f"{features['last_20_percent_volume']:.4f}")
        st.metric("最後20%低下率", f"{features['last_20_percent_drop_rate']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

def display_comprehensive_analysis_with_context(features):
    """コンテキスト考慮の総合分析表示"""
    st.markdown('<h3 class="sub-header">コンテキスト考慮総合分析</h3>', unsafe_allow_html=True)
    
    # ルールベースの評価（コンテキスト考慮）
    rule_based_evaluation = evaluate_clarity(features)
    
    # 機械学習による評価
    ml_available = st.session_state.model_trained
    
    if ml_available:
        try:
            ml_prediction, ml_confidence = st.session_state.ml_model.predict(features)
            ml_success = True
        except Exception as ml_error:
            ml_prediction, ml_confidence = None, 0
            ml_success = False
            st.error(f"AI分析エラー: {ml_error}")
    else:
        ml_prediction, ml_confidence = None, 0
        ml_success = False
    
    # 結果表示
    if ml_success and ml_available:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 AI分析結果")
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            if ml_prediction == "良好":
                st.success(f"**予測結果: {ml_prediction}**")
                advice_text = f"良い発話です！{features.get('conversation_context', '')}レベルの会話として語尾まで伝わりやすい話し方です。"
            elif ml_prediction == "文末が弱い":
                st.warning(f"**予測結果: {ml_prediction}**")
                advice_text = f"{features.get('conversation_context', '')}レベルの会話でも、文末を少し意識すると良いでしょう。"
            else:
                st.info(f"**予測結果: {ml_prediction}**")
                advice_text = f"さらなる改善の余地があります。"
            
            st.metric("予測信頼度", f"{ml_confidence:.1%}")
            st.write("**AIのアドバイス:**")
            st.write(advice_text)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### コンテキスト考慮分析")
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            if rule_based_evaluation['clarity_level'] == "良好":
                st.success(f"**評価: {rule_based_evaluation['clarity_level']}**")
            else:
                st.warning(f"**評価: {rule_based_evaluation['clarity_level']}**")
                
            st.metric("明瞭度スコア", f"{rule_based_evaluation['score']}/100")
            st.write("**コンテキスト考慮アドバイス:**")
            st.write(rule_based_evaluation['advice'])
            st.write(f"**分析コンテキスト**: {rule_based_evaluation.get('recording_context', '不明')}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # ルールベースのみの結果表示
        st.markdown("#### コンテキスト考慮分析結果")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("明瞭度スコア", f"{rule_based_evaluation['score']}/100")
            if rule_based_evaluation['clarity_level'] == "良好":
                st.success(f"**評価: {rule_based_evaluation['clarity_level']}**")
            else:
                st.warning(f"**評価: {rule_based_evaluation['clarity_level']}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("アドバイス")
            st.write(rule_based_evaluation['advice'])
            st.write(f"**コンテキスト**: {rule_based_evaluation.get('recording_context', '不明')}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 練習のヒント（コンテキスト考慮）
    show_context_aware_practice_hints(features, rule_based_evaluation)

def show_context_aware_practice_hints(features, evaluation):
    """コンテキスト考慮の練習ヒント"""
    st.markdown('<h3 class="sub-header">録音環境を考慮した練習のヒント</h3>', unsafe_allow_html=True)
    
    recording_source = features.get('recording_source', 'file')
    conversation_context = features.get('conversation_context', 'casual')
    
    with st.expander("録音環境別の改善方法", expanded=False):
        
        if recording_source == "microphone":
            st.markdown("""
            **マイク録音での練習について**
            - マイクは実際の会話より音量を増幅するため、より厳しい基準で評価しています
            - 実際の親密な会話では、この練習結果より少し声を意識する必要があります
            - マイクから適切な距離（20-30cm）を保って録音してください
            """)
        elif recording_source == "smartphone":
            st.markdown("""
            **スマートフォン録音での練習について**
            - スマートフォンのマイクは内蔵AGC（自動音量調整）が働く場合があります
            - 静かな環境で録音してください
            - 口元から15-20cm程度の距離で録音いただくことを推奨します
            """)
        
        if conversation_context in ["intimate", "very_intimate"]:
            st.markdown("""
            **親密な会話での注意点**
            - 小声でも語尾の明瞭さは重要です
            - 家族や恋人との会話でも、重要な情報は最後まで伝える意識を持ちましょう
            - 普段より少しだけ語尾を意識するだけで大きく改善されます
            """)
        
        if evaluation['clarity_level'] != "良好":
            st.markdown(f"""
            ** {conversation_context}レベルでの改善方法**
            1. **呼吸を意識する**: 話し始める前に十分な息を吸う
            2. **文末を1音上げる**: 最後の単語を意識的に少し強調
            3. **短い文で区切る**: 長い文は途中で息が切れやすい
            4. **録音して客観視**: 自分の声を客観的に聞く習慣をつける
            """)
        else:
            st.markdown("""
            **現在の良い話し方を維持**
            - この調子で親密な会話を続けましょう
            - 他のシチュエーションでも同様に意識してみてください
            - 疲れているときや急いでいるときも語尾を忘れずに
            """)
    
    st.success("コンテキストを考慮した分析が完了しました！日常会話での実践を心がけましょう。")    

# ＃以下補正情報追加以前のデータ            

    page = st.session_state.page  # 追加: page変数をセッションから取得

    if page == "モデル訓練":
        st.markdown('<h2 class="sub-header">AI訓練と評価</h2>', unsafe_allow_html=True)
        
        # モデル訓練の説明
        st.markdown("""
        <div class="info-box">
        <h3>🤖 AIについて</h3>
        <p>このページでは、機械学習モデル（AI）を訓練・評価することができます。</p>
        <p>AIを訓練することで、音声分析の精度が向上し、より詳細なフィードバックが得られます。</p>
        <p><strong>※ 初回利用時は必ずAI訓練を実行してください。</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # 訓練前後の状態表示
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.model_trained:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.success("**AI訓練完了**")
                st.write("AIが訓練されています。高精度な分析が利用可能です。")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.warning("**AIは未訓練**")
                st.write("AIがまだ訓練されていません。下のボタンから訓練を開始してください。")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.info("**訓練について**")
            st.write("- 訓練は1-2分程度で完了します")
            st.write("- シミュレーションデータを使用")
            st.write("- 訓練後は練習ページで高精度分析が可能")
            st.markdown('</div>', unsafe_allow_html=True)

        # 訓練ボタンとオプション
        if st.session_state.model_trained:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("AIモデルの再訓練", type="secondary"):
                    st.session_state.model_trained = False
                    st.rerun()
            with col2:
                if st.button("訓練済みモデル詳細"):
                    # 既存モデルの詳細表示
                    importance = st.session_state.ml_model.get_feature_importance()
                    if importance:
                        st.subheader("特徴量の重要度")
                        importance_df = pd.DataFrame(
                            list(importance.items()), 
                            columns=['特徴量', '重要度']
                        ).sort_values('重要度', ascending=False)

                        # グラフ表示
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['特徴量'], importance_df['重要度'])
                        ax.set_xlabel('重要度')
                        ax.set_title('各特徴量がAI予測に与える影響')
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.write("---")
            st.markdown('<h3 class="sub-header">AI訓練を開始</h3>', unsafe_allow_html=True)
            
            if st.button("🤖 AI訓練を開始", type="primary"):
                # モデル訓練の詳細実行
                with st.spinner("AIを訓練中..."):
                    # プログレスバーとステータス
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # ステップ1: データ生成
                    status_text.text("ステップ 1/4: 訓練データ生成中...")
                    progress_bar.progress(25)
                    X, y = generate_training_data()
                    time.sleep(0.5)

                    # データの詳細表示
                    st.markdown('<h4 class="sub-header">生成されたデータの詳細</h4>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("総サンプル数", len(X))
                    with col2:
                        st.metric("特徴量の数", X.shape[1])
                    with col3:
                        unique, counts = np.unique(y, return_counts=True)
                        st.write("**クラス分布:**")
                        for label, count in zip(unique, counts):
                            st.write(f"- {label}: {count}個")

                    # ステップ2: データ前処理
                    status_text.text("ステップ 2/4: データ前処理中...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
            
                    # ステップ3: モデル訓練
                    status_text.text("ステップ 3/4: AI学習中...")
                    progress_bar.progress(75)

                    # 実際の訓練実行
                    if st.session_state.ml_model.train(X, y):
                        st.session_state.model_trained = True

                        # ステップ4: 結果分析
                        status_text.text("ステップ 4/4: 結果分析中...")
                        progress_bar.progress(100)
                        time.sleep(0.5)

                        st.success("AI訓練が完了しました！")

                        # 特徴量重要度の表示
                        importance = st.session_state.ml_model.get_feature_importance()
                        if importance:
                            st.markdown('<h4 class="sub-header">特徴量の重要度（グラフ）</h4>', unsafe_allow_html=True)
                            importance_df = pd.DataFrame(
                                list(importance.items()), 
                                columns=['特徴量', '重要度']
                            ).sort_values('重要度', ascending=False)

                            # 横棒グラフで表示
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(importance_df['特徴量'], importance_df['重要度'])
                            ax.set_xlabel('重要度')
                            ax.set_title('各特徴量がAI予測に与える影響度')
                            plt.tight_layout()
                            st.pyplot(fig)

                            # 結果の解釈
                            st.markdown('<h4 class="sub-header">結果の解釈</h4>', unsafe_allow_html=True)
                            top_feature = importance_df.iloc[0]['特徴量']
                            st.markdown(f"""
                            <div class="info-box">
                            <p><strong>最も重要な特徴量</strong>: {top_feature}</p>
                            <p>この特徴量がAIの予測に最も大きく影響しています。</p>
                            <p>音声の品質を判断する上で{top_feature}が最も重要な要素だということを
                            AIが学習したことを示します。</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                    else:
                        st.error("AI訓練に失敗しました")                

                    # プログレスバーをクリア
                    progress_bar.empty()
                    status_text.empty()
                    
                    # 次のステップ案内
                    if st.session_state.model_trained:
                        st.markdown("""
                        <div class="next-step">
                        <h4>次のステップ</h4>
                        <p>AI訓練が完了しました！「練習を始める」ページで高精度な音声分析を試してみましょう。</p>
                        </div>
                        """, unsafe_allow_html=True)

        # 次のステップガイド
        show_next_step_guide()



    # アプリケーション終了時のクリーンアップ処理
    if st.session_state.get('temp_audio_file') and os.path.exists(st.session_state.temp_audio_file):
        try:
            os.unlink(st.session_state.temp_audio_file)
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")

# アプリの実行
if __name__ == "__main__":
    try:
        main()
    except Exception as app_error:
        st.error(f"アプリケーションでエラーが発生しました: {app_error}")
        logger.error(f"アプリケーションエラー: {app_error}", exc_info=True)
    finally:
        # セッション状態のクリア
        st.session_state.clear()
        st.session_state.first_visit = True
        st.session_state.practice_count = 0
        st.session_state.model_trained = False
        st.session_state.show_guide = True
        st.session_state.end_of_sentence_detected = False
        st.session_state.current_drop_rate = 0.0    