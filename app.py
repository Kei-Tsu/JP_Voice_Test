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
    st.sidebar.warning("streamlit-webrtcライブラリがインストールされていません。'pip install streamlit-webrtc'でインストールしてください。")
    # ダミー関数を定義（エラーを防ぐため）
    def webrtc_streamer(*args, **kwargs):
        return None
    class WebRtcMode:
        SENDONLY = "sendonly"
    class RTCConfiguration:
        def __init__(self, *args, **kwargs):
            pass

import av
import scipy.io.wavfile

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
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = VoiceQualityModel()  # 音声品質モデルのインスタンス
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False  # モデルの訓練状態


# セッション状態の拡張(リアルタイム機能用)    
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
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
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

# 音声フレームを処理するコールバック関数（簡略版）
def audio_frame_callback(frame):
    """音声フレームを処理するコールバック関数"""
    try:
        # フレームをnumpy配列に変換
        sound = frame.to_ndarray()
        
        # 現在の音量レベルを計算
        if len(sound) > 0:
            audio_data = sound.flatten()
            rms = np.sqrt(np.mean(audio_data**2))
            db = 20 * np.log10(max(rms, 1e-10))
            
            # セッション状態に音量履歴を追加
            st.session_state.volume_history.append({"音量": db})
            if len(st.session_state.volume_history) > 100:
                st.session_state.volume_history.pop(0)
            
            # パラメータ取得
            silence_threshold = st.session_state.get('silence_threshold', -40)
            min_silence_duration = st.session_state.get('min_silence_duration', 500)
            
            # 音量判定と処理
            if db > silence_threshold:
                st.session_state.last_sound_time = time.time()
                st.session_state.end_of_sentence_detected = False
            else:
                # 無音状態の処理
                current_time = time.time()
                silence_duration = (current_time - st.session_state.last_sound_time) * 1000
                
                if silence_duration > min_silence_duration and not st.session_state.end_of_sentence_detected:
                    st.session_state.end_of_sentence_detected = True
                    
                    # 文末の音量低下率を計算
                    if len(st.session_state.volume_history) > 10:
                        recent_volumes = [item["音量"] for item in st.session_state.volume_history[-10:]]
                        
                        if len(recent_volumes) > 5:
                            before_avg = sum(recent_volumes[-7:-4]) / 3
                            after_avg = sum(recent_volumes[-3:]) / 3
                            drop_rate = (before_avg - after_avg) / (abs(before_avg) + 1e-10)
                            
                            st.session_state.current_drop_rate = drop_rate
                            
                            # フィードバック履歴に追加
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

def main():
    # 特徴抽出器の初期化
    feature_extractor = VoiceFeatureExtractor()

    # アプリのタイトルと説明
    st.title('語尾までしっかりマスター')
    st.write('身近な会話をしっかり伝えることで、大切な人とのコミュニケーションを高めよう')

    # サイドバーでナビゲーション
    page = st.sidebar.selectbox("ページ選択", ["ホーム", "練習を始める", "モデル訓練"])
    st.session_state.page = page

    # サイドバーに設定を追加（リアルタイム機能使用時のみ）
    if page == "練習を始める" and WEBRTC_AVAILABLE:
        st.sidebar.title("リアルタイム評価設定")
        st.session_state.silence_threshold = st.sidebar.slider(
            "無音しきい値 (dB)", 
            -80, 0, -40,
            help="音声を「無音」と判断する音量レベルを設定します。"
        )
        st.session_state.min_silence_duration = st.sidebar.slider(
            "最小無音時間 (ms)", 
            100, 500, 300,
            help="この時間以上の無音が続いた場合に「無音区間」と判断します。"
        )

    # ページごとの表示内容
    if page == "ホーム":
        st.markdown('<h1 class="main-header">Welcome！</h1>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("""
        このアプリは日本語の短い会話を分析し、文末の明瞭さを高めることに注力しています。
        日本語は言語の特徴上、自然と語尾の音声量が低下しがちです。
        あなたの発話を分析して、話し方を高めるヒントを提供します。
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "練習を始める":
        st.markdown('<h1 class="sub-header">音声練習</h1>', unsafe_allow_html=True)
    
        # カテゴリーとサンプル文の選択
        category = st.selectbox("会話カテゴリーを選択", list(CONVERSATION_SAMPLES.keys()))
        sample_index = st.selectbox(
            "サンプル文を選択", 
            range(len(CONVERSATION_SAMPLES[category])),
            format_func=lambda i: CONVERSATION_SAMPLES[category][i]
        )
        
        selected_sample = CONVERSATION_SAMPLES[category][sample_index]
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("### 読み上げるサンプル文")
        st.write(selected_sample)
        st.write("このサンプル文を、普段のように自然に読み上げてください。")
        st.markdown('</div>', unsafe_allow_html=True)

        # 練習方法の選択
        if WEBRTC_AVAILABLE:
            practice_method = st.radio("練習方法を選択", ["音声ファイルをアップロード", "リアルタイム評価"])
        else:
            practice_method = st.radio("練習方法を選択", ["音声ファイルをアップロード"])
            st.info("リアルタイム評価を使用するには、streamlit-webrtcをインストールしてください。")

        if practice_method == "音声ファイルをアップロード":
            uploaded_file = st.file_uploader(
                "音声ファイルをアップロードしてください", 
                type=["wav", "mp3"],
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                tmp_file_path = None  # 変数を事前に初期化
                try:
                    # 一時ファイルとして保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # 音声ファイルを再生可能に表示
                    st.audio(tmp_file_path, format='audio/wav')

                    # 音声データの読み込み
                    y, sr = librosa.load(tmp_file_path, sr=None)

                    # 音声特徴量の抽出
                    features = feature_extractor.extract_features(y, sr)
                    
                    # 音声分析の視覚化
                    st.subheader("音声分析結果")
                    fig = plot_audio_analysis(features, y, sr)
                    st.pyplot(fig)
                    
                    # 音量分析結果の表示
                    st.markdown('<h2 class="sub-header">音量分析結果</h2>', unsafe_allow_html=True)
                            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("平均音量", f"{features['mean_volume']:.4f}")
                        st.metric("文頭音量", f"{features['start_volume']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("文中音量", f"{features['middle_volume']:.4f}")
                        st.metric("文末音量", f"{features['end_volume']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("文末音量低下率", f"{features['end_drop_rate']:.4f}")
                        st.metric("文末音量低下率(最後の20%)", f"{features['last_20_percent_drop_rate']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # 音声明瞭度評価
                    evaluation = evaluate_clarity(features)
                    
                    st.markdown('<h2 class="sub-header">音声明瞭度評価</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("明瞭度スコア", f"{evaluation['score']}/100")
                        st.metric("明瞭度評価", evaluation['clarity_level'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.write("アドバイス")
                        st.write(evaluation['advice'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    # 一時ファイルを削除
                    os.unlink(tmp_file_path)

                except Exception as e:
                    st.error(f"音声分析中にエラーが発生しました: {e}")
                finally:
                    # 一時ファイルの削除
                    if tmp_file_path is not None and os.path.exists(tmp_file_path):
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            logger.error(f"一時ファイル削除エラー: {e}")
                            
        elif practice_method == "リアルタイム評価" and WEBRTC_AVAILABLE:
            st.write("### リアルタイム評価")
            st.info("「START」ボタンをクリックし、ブラウザからのマイク使用許可を承認してください。")

            # プレースホルダーの準備
            status_placeholder = st.empty()
            volume_placeholder = st.empty()
            feedback_placeholder = st.empty()

            try:
                # WebRTCストリーマーを設定
                webrtc_ctx = webrtc_streamer(
                    key="speech-evaluation",
                    mode=WebRtcMode.SENDONLY,
                    audio_frame_callback=audio_frame_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": False, "audio": True}
                )

                # WebRTC接続が有効な場合
                if webrtc_ctx.state.playing:
                    # 音量メーターの表示
                    display_volume_meter(volume_placeholder)
                            
                    # 状態表示
                    if st.session_state.end_of_sentence_detected:
                        drop_rate = st.session_state.current_drop_rate
                                
                        if drop_rate < 0.1:
                            status_placeholder.success("良い感じです！語尾までしっかり発音できています。")
                        elif drop_rate < 0.25:
                            status_placeholder.info("語尾がやや弱まっています。もう少し意識しましょう。")
                        else:
                            status_placeholder.warning("語尾の音量が大きく低下しています。文末を意識して！")
                    else:
                        status_placeholder.info("マイクに向かってサンプル文を読み上げてください。")
                            
                    # フィードバック履歴の表示
                    display_feedback_history(feedback_placeholder)
                else:
                    status_placeholder.warning("マイク接続待機中...「START」ボタンをクリックしてください。")

            except Exception as e:
                st.error(f"リアルタイム機能でエラーが発生しました: {e}")

    # モデル訓練ページ
    elif page == "モデル訓練":
        st.markdown('<h1 class="sub-header">モデル訓練と評価</h1>', unsafe_allow_html=True)
        
        st.write("""
        このページでは、機械学習モデルを訓練・評価することができます。
        サンプルデータを使用してモデルを訓練します。
        """)    
       
        if st.button("モデル訓練"):
            with st.spinner("モデルを訓練中..."):
                # シミュレーションデータの生成
                X, y = generate_training_data()

                # モデルの初期化と訓練
                if st.session_state.ml_model.train(X, y):
                    st.session_state.model_trained = True
                    st.success("モデルの訓練が完了しました")

                    # 特徴量の重要度を表示
                    importance = st.session_state.ml_model.get_feature_importance()
                    if importance:
                        st.subheader("特徴量の重要度")
                        importance_df = pd.DataFrame(
                            list(importance.items()), 
                            columns=['特徴量', '重要度']
                        ).sort_values('重要度', ascending=False)
                        st.bar_chart(importance_df.set_index('特徴量'))
                else:
                    st.error("モデルの訓練に失敗しました")

# アプリの実行
if __name__ == "__main__":
    main()  
