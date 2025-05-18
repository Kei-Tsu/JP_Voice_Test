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

#追加するインポート文
from ml_model import VoiceQualityModel, generate_training_data, create_dataset_from_files

# 機械学習関連のインポートを追加（ファイル上部）
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
    
# FFmpeg警告の無視設定を追加
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

# WebRTC関連のライブラリのインポート
# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration  # ブラウザで音声を録音するためのライブラリ
except ImportError:
    st.error("streamlit-webrtcライブラリがインストールされていません。'pip install streamlit-webrtc'でインストールしてください。")
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
import logging

#ロガーの設定
logger = logging.getLogger(__name__)

# FFmpeg警告の無視設定を追加（ここに追加）
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

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


# セッション状態の拡張    
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


# アプリケーション設定
st.set_page_config(
    page_title="語尾までしっかりマスター",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（アプリの見栄えをよりよいものにするため）
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
        
        for i, feedback in enumerate(reversed(st.session_state.feedback_history[-5:])):  # 最新5件のみ表示
            level = feedback["level"]
            css_class = f"feedback-box feedback-{level}"
            
            placeholder.markdown(
                f"<div class='{css_class}'>"
                f"<p>{feedback['time']} - {feedback['emoji']} {feedback['message']}</p>"
                f"<p>文末音量低下率: {feedback['drop_rate']:.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

# 文末音量低下率に基づくフィードバックを返す関数
def get_feedback(drop_rate):
    """
    drop_rateに応じてフィードバック（レベル・メッセージ・絵文字）を返す
    """
    if drop_rate < 0.1:
        return {
            "level": "good",
            "message": "良い感じです！語尾までしっかり発音できています。",
            "emoji": "✅"
        }
    elif drop_rate < 0.25:
        return {
            "level": "medium",
            "message": "語尾がやや弱まっています。もう少し意識しましょう。",
            "emoji": "🟡"
        }
    else:
        return {
            "level": "bad",
            "message": "語尾の音量が大きく低下しています。文末を意識して！",
            "emoji": "⚠️"
        }

# 録音開始/停止ボタン
def toggle_recording():
    if st.session_state.recording:
        st.toast(f"**録音停止**", icon="🎤")
    else:
        st.toast(f"**録音開始**", icon="🎤")
    st.session_state.recording = not st.session_state.recording
    if not st.session_state.recording:
        # 録音停止時、キャプチャも停止
        st.session_state.is_capturing = False

# 音声フレームを処理するコールバック関数(非同期処理対応バージョン)
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
            try:
                st.session_state.volume_history.append({"音量": db})
                if len(st.session_state.volume_history) > 100:
                    st.session_state.volume_history.pop(0)
            except Exception as volume_error:
                logger.error(f"音量履歴の更新エラー: {volume_error}")
          
   
            # パラメータ取得（サイドバーから設定可能）
            try:
                silence_threshold = st.session_state.get('silence_threshold', -40)  # 無音判定の閾値（dB）
                min_silence_duration = st.session_state.get('min_silence_duration', 500)  # 最小無音時間（ms）    
                       
                # 音量判定と処理（音量が閾値より大きい場合、音声あり）
                if db > silence_threshold:
                    st.session_state.last_sound_time = time.time()
                    st.session_state.end_of_sentence_detected = False
                
                    # 録音開始判定（録音モードがオンで、かつキャプチャが開始されていない場合）
                    try:
                        if st.session_state.recording and not st.session_state.is_capturing:
                            st.session_state.is_capturing = True
                            if st.session_state.capture_buffer is None:
                                st.session_state.capture_buffer = AudioSegment.empty()
                    except Exception as rec_error:
                        logger.error(f"録音開始処理エラー: {rec_error}")
                else:
                    # 無音状態の処理
                    try:
                        # 無音状態が一定時間続いた場合、文末と判断
                        current_time = time.time()
                        silence_duration = (current_time - st.session_state.last_sound_time) * 1000  # ミリ秒に変換
                
                        if silence_duration > min_silence_duration and not st.session_state.end_of_sentence_detected:
                            st.session_state.end_of_sentence_detected = True                   
                         
                            # 文末の音量低下率を計算
                            if len(st.session_state.volume_history) > 10:
                                try:
                                    recent_volumes = [item["音量"] for item in st.session_state.volume_history[-10:]]
                        
                                    # 簡易的な文末判定
                                    if len(recent_volumes) > 5:
                                        before_avg = sum(recent_volumes[-7:-4]) / 3  # 文末前の平均
                                        after_avg = sum(recent_volumes[-3:]) / 3    # 文末の平均
                                        drop_rate = (before_avg - after_avg) / (abs(before_avg) + 1e-10)
                            
                                        # 判定結果をセッション状態に保存
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
                                except Exception as fb_error:
                                    logger.error(f"フィードバック生成エラー: {fb_error}")

                        #　録音停止判定（録音中かつ無音が続く場合）
                        auto_stop_duration = st.session_state.get('auto_stop_duration', 1000)
                        if st.session_state.recording and st.session_state.is_capturing and silence_duration > auto_stop_duration:
                            st.session_state.is_capturing = False
                            # この時点で録音を保存する処理を呼び出す（非同期処理するため、直接呼び出さない）

                        #キャプチャー処理           
                        try:
                            # キャプチャー中であれば音声データを追加
                            if st.session_state.recording and st.session_state.is_capturing:
                                # 音声フレームからpydub形式に変換
                                audio_segment = AudioSegment(
                                    data=frame.to_ndarray().tobytes(),
                                    sample_width=frame.format.bytes,
                                    frame_rate=frame.sample_rate,
                                    channels=len(frame.layout.channels),
                                )
                                 
                                #キャプチャバッファに追加
                                if st.session_state.capture_buffer is None:
                                    st.session_state.capture_buffer = audio_segment
                                else:
                                    st.session_state.capture_buffer += audio_segment
                        except Exception as processing_error:
                            logger.error(f"音声処理エラー: {processing_error}")
                    except Exception as e:
                        logger.error(f"音声フレーム処理エラー: {e}", exc_info=True)
            except Exception as param_error:
                logger.error(f"パラメータ取得エラー: {param_error}")
    except Exception as e:
        logger.error(f"音声フレーム処理エラー: {e}", exc_info=True)        

        return frame
            
              
        #　録音停止判定（録音中かつ無音が続く場合）
        auto_stop_duration = st.session_state.get('auto_stop_duration', 1000)
        if st.session_state.recording and st.session_state.is_capturing and silence_duration > auto_stop_duration:
            st.session_state.is_capturing = False
            # この時点で録音を保存する処理を呼び出す（非同期処理するため、直接呼び出さない）
           
            try:
                # キャプチャー中であれば音声データを追加
                if st.session_state.recording and st.session_state.is_capturing:
                # 音声フレームからpydub形式に変換
                    audio_segment = AudioSegment(
                        data=frame.to_ndarray().tobytes(),
                        sample_width=frame.format.bytes,
                        frame_rate=frame.sample_rate,
                        channels=len(frame.layout.channels),
                    )
            except Exception as capture_error:
                logger.error(f"音声キャプチャエラー: {capture_error}")

                # キャプチャバッファに追加
                if st.session_state.capture_buffer is None:
                    st.session_state.capture_buffer = audio_segment
                else:
                    st.session_state.capture_buffer += audio_segment
    
    except Exception as e:
        logger.error (f"音声フレーム処理エラー: {e}", exc_info = True)
    
    return frame

# 一時ファイルへの保存とオーディオプレーヤーの表示 (非同期関数)
async def save_and_analyze_audio(audio_segment):
    if audio_segment is None or len(audio_segment) == 0:
        return
    
    # 最低録音時間のチェック
    min_recording_duration = st.session_state.get('min_recording_duration', 2)
    recording_duration = len(audio_segment) / 1000.0  # ミリ秒から秒に変換
    
    if recording_duration < min_recording_duration:
        return

 # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        temp_file_path = tmp_file.name
    
    # 音声ファイルの保存 (非同期処理)
    await asyncio.to_thread(audio_segment.export, temp_file_path, format="wav")

    # 以前の一時ファイルがあれば削除
    if st.session_state.temp_audio_file and os.path.exists(st.session_state.temp_audio_file):
        try:
            os.unlink(st.session_state.temp_audio_file)
        except Exception as e:
            logger.warning(f"一時ファイルの削除に失敗: {e}")

    # 新しい一時ファイルのパスを保存
    st.session_state.temp_audio_file = temp_file_path
    st.session_state.recorded_audio = audio_segment
    
    # 処理後にバッファをクリア
    st.session_state.capture_buffer = None
    
    # GCを強制的に実行してメモリを解放
    import gc
    gc.collect()

    # 音声分析を行う
    try:
        # 音声データの読み込み
        y, sr = librosa.load(temp_file_path, sr=None)
        
        # 特徴量抽出器の初期化
        feature_extractor = VoiceFeatureExtractor()
        
        # 音声特徴量の抽出
        features = feature_extractor.extract_features(y, sr)
        
        # 評価結果の生成
        evaluation = evaluate_clarity(features)
        
        # セッション状態に結果を保存
        st.session_state.last_analysis = {
            "features": features,
            "evaluation": evaluation,
            "audio_path": temp_file_path,
            "audio_data": y,
            "sr": sr
        }
        
        # 分析完了のフラグをセット
        st.session_state.analysis_complete = True
 
    except Exception as e:
        logger.error(f"音声分析中にエラーが発生しました: {e}", exc_info=True)
        st.session_state.analysis_error = str(e)

# 非同期関数を実行するためのヘルパー関数
def run_async(async_func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        loop.close()


# VoiceFeatureExtractorのインポートまたは定義
from ml_model import VoiceFeatureExtractor

# 明瞭度評価関数の定義
def evaluate_clarity(features):
    """
    音声特徴量から明瞭度スコアと評価を返す簡易関数
    """
    # 例: 文末音量低下率を主な指標とする
    drop_rate = features.get("end_drop_rate", 0.0)
    mean_volume = features.get("mean_volume", -40.0)
    score = max(0, 100 - int(drop_rate * 100) - int(abs(mean_volume + 20)))  # 仮のスコア計算

    if drop_rate < 0.1 and mean_volume > -30:
        clarity_level = "良好"
        advice = "発話は良好です！文末まで明瞭に話せています。"
    elif drop_rate < 0.25:
        clarity_level = "普通"
        advice = "文末がやや弱くなっています。文末まで意識して話しましょう。"
    elif drop_rate < 0.4:
        clarity_level = "やや弱い"
        advice = "文末の音量が低下しています。息を残して最後まで話しましょう。"
    else:
        clarity_level = "弱い"
        advice = "文末がかなり弱くなっています。文末を意識して練習しましょう。"

    return {
        "score": score,
        "clarity_level": clarity_level,
        "advice": advice
    }

# アプリのメイン部分
def main():
    # 特徴抽出器の初期化
    feature_extractor = VoiceFeatureExtractor()

     # 機械学習モデルの初期化と訓練
    if not st.session_state.model_trained:
        with st.spinner("機械学習モデルを初期化中..."):
            try:
                # シミュレーションデータの生成
                X, y = generate_training_data()
                
                # モデルの訓練
                if st.session_state.ml_model.train(X, y):
                    st.session_state.model_trained = True
                    st.toast("モデルの初期化が完了しました")
            except Exception as e:
                st.error(f"モデル初期化エラー: {e}")
    
    # アプリのタイトルと説明
    st.title('語尾までしっかりマスター')
    st.write('身近な会話をしっかり伝えることで、大切な人とのコミュニケーションを高めよう')

    # サイドバーでナビゲーション
    page = st.sidebar.selectbox("ページ選択", ["ホーム", "練習を始める","モデル訓練", "本アプリについて"])
    st.session_state.page = page  # ページ状態を更新

    # サイドバーに無音検出設定を追加
    if page == "練習を始める":
        st.sidebar.title("無音検出設定")
        st.session_state.silence_threshold = st.sidebar.slider(
            "無音しきい値 (dB)", 
            -80, 0, -40,
            help="音声を「無音」と判断する音量レベルを設定します。\n"
                "値が小さいほど（例：-50dB）より小さな音も「音声あり」と判断します。\n"
                "値が大きいほど（例：-20dB）大きな音のみを「音声あり」と判断します。"
        )

        st.session_state.min_silence_duration = st.sidebar.slider(
            "最小無音時間 (ms)", 
            100, 500, 300,
            help="この時間以上の無音が続いた場合に「無音区間」と判断します。\n"
                "短すぎると話の途中の短い間も無音と判断され、\n"
                "長すぎると長めの間も音声の一部と判断されます。"
        )

        st.sidebar.title("録音設定")
        st.session_state.auto_stop_duration = st.sidebar.slider(
            "無音検出時の自動停止 (ms)", 
            100, 2000, 1000,
            help="この時間以上の無音が続くと、自動的に録音を停止します。\n"
                "話者の発話が終わったことを検出するための設定です。\n"
                "短すぎると話の途中で録音が止まり、長すぎると無駄な無音時間が録音されます。"
        )
        
        st.session_state.min_recording_duration = st.sidebar.slider(
            "最低録音時間 (秒)", 
            1, 10, 2,
            help="録音を保存する最低限の長さを設定します。\n"
                "これより短い録音は無視されます。\n"
                "短すぎると雑音なども録音されやすく、長すぎると短い返事なども無視されます。"
        )

    # ページごとの表示内容
    if page == "ホーム":
        st.markdown('<h1 class="main-header">Welcome！</h1>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("""
        このアプリは日本語の短い会話を分析し、文末の明瞭さを高めることに注力しています。
        日本語は言語の特徴上、自然と語尾の音声量が低下しがちです。
        家族や近しい人とのカジュアルな会話や、小声の会話で特にこの傾向が見られます。
        一方で意識しすぎて大きすぎたり、力を入れすぎるとコミュニケーションにマイナスです。
        あなたの発話を分析して、話し方を高めるヒントを提供します。
        ぜひ、あなたの声を聞かせてください。            
        
        このアプリでは、あなたの発話を分析して、語尾の明瞭さを評価し、改善のためのヒントを
        提供します。2つの方法で練習できます：
        
        1. **録音済みの音声ファイルをアップロード**して詳細な分析を受ける
        2. **リアルタイム評価**でマイクから話しながら即時フィードバックを得る
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown('<h2 class="sub-header">使い方</h2>', unsafe_allow_html=True)
        st.write("1. 左のサイドバーから「練習を始める」を選択")
        st.write("2. 練習したいサンプル文を選んで読み上げる")
        st.write("3. 「音声ファイルをアップロード」または「リアルタイム評価」を選択")
        st.write("4. 分析結果とアドバイスを確認")
    
        if st.button("練習を始める"):
            st.session_state.page = "練習を始める"
            st.rerun() 

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
        practice_method = st.radio("練習方法を選択", ["音声ファイルをアップロード", "リアルタイム評価"])

        if practice_method == "音声ファイルをアップロード":
            # ファイルをアップロードする機能
            uploaded_file = st.file_uploader(
                "音声ファイルをアップロードしてください", 
                type=["wav", "mp3"],
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    tmp_file_path = None  # 初期値を設定

                    # 一時ファイルとして保存
                    tmp_file_path = None               
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                except Exception as e:
                        st.error(f"ファイル保存中にエラーが発生しました: {e}")
                        st.stop()

                # 音声ファイルの長さをチェック
                audio_length = librosa.get_duration(path=tmp_file_path)

                # 音声ファイルを再生可能に表示
                st.audio(tmp_file_path, format='audio/wav')

                # 音声データの読み込み
                y, sr = librosa.load(tmp_file_path, sr=None)

                # 音声特徴量の抽出
                features = feature_extractor.extract_features(y, sr)
                
                # 音声分析の視覚化
                st.subheader("音声分析結果")    

                # plot_audio_analysis関数の定義
                def plot_audio_analysis(features, y, sr):
                    import matplotlib.pyplot as plt
                    import numpy as np

                    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                    # 波形の描画
                    times = np.arange(len(y)) / sr
                    ax[0].plot(times, y, color='b')
                    ax[0].set_title("音声波形")
                    ax[0].set_ylabel("振幅")
                    ax[0].grid(True)

                    # 音量(RMS)の描画
                    if "rms" in features and "times" in features:
                        ax[1].plot(features["times"], features["rms"], color='g')
                        ax[1].set_title("音量 (RMS)")
                        ax[1].set_ylabel("RMS")
                        ax[1].set_xlabel("時間 (秒)")
                        ax[1].grid(True)
                    else:
                        ax[1].text(0.5, 0.5, "RMSデータなし", ha='center', va='center')

                    plt.tight_layout()
                    return fig

                fig = plot_audio_analysis(features, y, sr)
                st.pyplot(fig)
                
                # 音量分析結果の表示
                st.markdown('<h2 class="sub-header">音量分析結果</h2>', unsafe_allow_html=True)
                        
                col1, col2, col3 = st.columns(3)
                with col1:
                    # フィードバックの表示
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
                    st.metric("スコア", f"{evaluation['score']}点")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write("アドバイス")
                    st.write(evaluation['advice'])
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 詳細な特徴量表示（オプション）
                    if st.checkbox("詳細な特徴量を表示"):
                        st.subheader("詳細な特徴量")
        
                    # 特徴量をカテゴリー別に整理して表示
                    # リスト型のデータを除外または変換
                    volume_features = {}
                    spectral_features = {}
                    rhythm_features = {}
        
                    # 音量関連特徴量
                    for k, v in features.items():
                        if 'volume' in k or 'drop' in k:
                        # rmsとtimesは除外（これらはリスト型のため）
                            if k not in ['rms', 'times'] and not isinstance(v, (list, np.ndarray)):
                                volume_features[k] = v
        
                    # スペクトル特徴量
                    for k, v in features.items():
                        if 'spectral' in k or 'mfcc' in k:
                            if not isinstance(v, (list, np.ndarray)):
                                spectral_features[k] = v
        
                    # リズム関連特徴量
                    for k, v in features.items():
                        if 'onset' in k or 'speech' in k:
                            if not isinstance(v, (list, np.ndarray)):
                                rhythm_features[k] = v
        
                    # データフレームに変換して表示
                    st.write("### 音量関連特徴量")
                    if volume_features:
                        volume_df = pd.DataFrame({
                            '特徴量': list(volume_features.keys()),
                            '値': list(volume_features.values())
                        })
                        st.dataframe(volume_df)
                    else:
                        st.write("表示できる音量関連特徴量がありません")
        
                    st.write("### スペクトル特徴量")
                    if spectral_features:
                        spectral_df = pd.DataFrame({
                            '特徴量': list(spectral_features.keys()),
                            '値': list(spectral_features.values())
                        })  
                        st.dataframe(spectral_df)
                    else:
                        st.write("表示できるスペクトル特徴量がありません")
        
                    st.write("### リズム関連特徴量")
                    if rhythm_features:
                        rhythm_df = pd.DataFrame({
                            '特徴量': list(rhythm_features.keys()),
                            '値': list(rhythm_features.values())
                        })
                        st.dataframe(rhythm_df)
                    else:
                        st.write("表示できるリズム関連特徴量がありません")

                         
                    # 練習のヒントと次のステップ
                    st.markdown('<h2 class="sub-header">練習のヒント</h2>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        
                    if evaluation["clarity_level"] in ["良好"]:
                        st.write("良い調子です！語尾まで発話できています。")
                        st.write("- この調子を維持してください！")
                        st.write("- 次のステップ: 他のサンプル文や自然な会話でも試してみましょう。")
                    elif evaluation["clarity_level"] in ["普通", "やや弱い"]:
                        st.write("- 文の最後まで息を残すように意識してみましょう。")
                        st.write("- 例えば、文末まで意識して話してみてください。")
                        st.write("- 次のステップ: 息を吸うタイミングを意識してみましょう。")
                    else:
                        st.write("- 文末を意識して、文を話し始める前に息を吸ってから話してみましょう。")
                        st.write("- 例えば、文末を1音上げるイメージで話してみましょう。")
                        st.write("- 次のステップ: 録音してご自身の声を聴くことで、話し方を確認しましょう。")
                        
                    st.markdown('</div>', unsafe_allow_html=True)


            #機械学習モデルによる予測
            if st.session_state.model_trained and 'features' in locals():
                try:
                    prediction,confidence = st.session_state.ml_model.predict(features)
                    
                    st.markdown('<h2 class="sub-header">機械学習モデルによる音声品質予測</h2>', unsafe_allow_html=True)
                           
                    col1, col2 = st.columns(2)
                           
                    with col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("音声品質", prediction)
                        st.metric("予測信頼度", f"{confidence:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.subheader("AIによるアドバイス")
            
                        if prediction == "良好":
                            st.success("発話は良好です！文末まで明瞭に話せています。")
                            st.write("この調子で続けましょう。")
                        elif prediction == "文末が弱い":
                             st.warning("文末の音量が低下しています。")
                             st.write("日本語は文末に重要情報が来ることが多いので、文末まで意識して話すと良いでしょう。")
                             st.write("- 息を深く吸ってから話し始める")
                             st.write("- 文の終わりまで十分な息を残しておく")
                             st.write("- 文末を少し強調する意識を持つ")
                        elif prediction == "小声すぎる":
                             st.warning("全体的に声が小さいです。")
                             st.write("相手に届くよう、もう少し声量を上げると良いでしょう。")
                             st.write("- 呼吸を少しだけ意識しましょう")
                             st.write("- 少し大きめの声を出す練習をする")
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    # 次のステップ: 録音してご自身の声を聴くことで、話し方を確認しましょう。
                    st.write("- 次のステップ: 録音してご自身の声を聴くことで、話し方を確認しましょう。")
                    st.markdown('</div>', unsafe_allow_html=True)    

                except Exception as e:
                    st.error(f"機械学習による予測中にエラーが発生しました: {e}")
            else:    
                if not st.session_state.model_trained:
                    st.info("機械学習モデルが初期化されていません。")
                elif 'features' not in locals():
                    st.info("音声特徴量が抽出されていません。音声を再度アップロードしてください")
                    
                    # 最後に一時ファイルを削除（必要に応じて）                   
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        try:
                            os.remove(tmp_file_path)
                        except Exception as e:
                            logger.warning(f"一時ファイルの削除に失敗: {e}") 
                       
            # エラー処理
            try:
                if 'e' in locals():
                    error_msg = str(e)

                    if "PySoundFile" in str(e):
                        st.error("音声ファイルの形式が正しくありません。別のwavまたはmp3形式のファイルをお試しください。")
                    elif "empty_file" in str(e):
                        st.error("アップロードがいるされた音声ファイルが空です。有効な音声ファイルをアップロードしてください。")
                    else:
                        st.error(f"音声分析中にエラーが発生しました: {error_msg}")
            except Exception:
                st.error("音声分析中にエラーが発生しました。")

            try:
                 os.unlink(tmp_file_path)
            except:
                pass
                    

        elif practice_method == "リアルタイム評価":
            st.write("### リアルタイム評価")
            st.info("「START」ボタンをクリックし、ブラウザからのマイク使用許可リクエストを承認してください。その後、サンプル文を読み上げると、リアルタイムで評価が表示されます。")

            # フォールバックオプションを追加
            use_fallback = st.checkbox("マイク接続に問題がある場合はチェック", False)
    
            if use_fallback:
                st.warning("簡易モードに切り替えます。ファイルアップロードを使用してください。")
                # 簡易モードでは何もしない、ユーザーは上のファイルアップロード機能を使用する
            
            else:
                # プレースホルダーの準備（動的更新用）
                status_placeholder = st.empty()
                volume_placeholder = st.empty()
                feedback_placeholder = st.empty()
                history_placeholder = st.empty()
                recording_status_placeholder = st.empty()
                analysis_placeholder = st.empty()

                try:
                    # WebRTCストリーマーを設定
                    webrtc_ctx = webrtc_streamer(
                        key="speech-evaluation",
                        mode=WebRtcMode.SENDONLY,
                        audio_frame_callback=audio_frame_callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"video": False, "audio": True}
                    )
                except Exception as e:
                    st.error(f"WebRTCストリーマーの設定中にエラーが発生しました: {e}")
                
                # WebRTC接続が有効な場合
                if webrtc_ctx.state.playing:
                    # 音量メーターの表示
                    display_volume_meter(volume_placeholder)
                            
                    # 状態表示
                    if st.session_state.end_of_sentence_detected:
                        drop_rate = st.session_state.current_drop_rate
                                
                        if drop_rate < 0.1:
                            status_placeholder.success("- 良い感じです！語尾までしっかり発音できています。")
                        elif drop_rate < 0.25:
                            status_placeholder.info("- 語尾がやや弱まっています。もう少し意識しましょう。")
                        else:
                            status_placeholder.warning("- 語尾の音量が大きく低下しています。文末を意識して！")
                    else:
                        status_placeholder.info("- マイクに向かってサンプル文を読み上げてください。文末を検出したらフィードバックを表示します。")
                            
                    # フィードバック履歴の表示
                    display_feedback_history(feedback_placeholder)

                    # 使い方の補足
                    with history_placeholder.expander("詳しい使い方"):
                        st.write("""
                        1. サンプル文を自然な声で読み上げてください
                        2. 一度に1つの文を読み、間を空けましょう
                        3. 文の終わりで少し間を空けると、文末と判断されフィードバックが表示されます
                        4. 複数の文を読んで練習を続けると、フィードバック履歴が表示されます
                        5. 音量メーターで自分の声の大きさを確認できます
                        """)
                                
                        st.write("### 音量レベルの目安")
                        st.write("- -20dB以上: 大きな声")
                        st.write("- -30dB～-20dB: 通常の会話音量")
                        st.write("- -40dB～-30dB: 小声")
                        st.write("- -40dB以下: 非常に小さいか聞こえない音量")
                else:
                    status_placeholder.warning("マイク接続待機中...「START」ボタンをクリックしてください。")

            # 非同期処理: キャプチャが完了したら音声を保存・分析
            
            try:
                if (st.session_state.get('capture_buffer') is not None and 
                    not st.session_state.is_capturing and 
                    st.session_state.end_of_sentence_detected):
                    # 非同期で音声を保存・分析
                    run_async(save_and_analyze_audio, st.session_state.capture_buffer)
                    # フラグをリセット
                    st.session_state.end_of_sentence_detected = False
          
            except Exception as e:
                st.error(f"マイク機能でエラーが発生しました: {e}")
                st.info("お使いのブラウザがWebRTCに対応していないか、マイクへのアクセス許可がない可能性があります。")

    elif page == "モデル訓練":
        # モデル訓練ページの内容
        st.markdown('<h1 class="sub-header">モデル訓練と評価</h1>', unsafe_allow_html=True)
        
        st.write("""
        このページでは、機械学習モデルを訓練・評価することができます。
        以下の方法でモデルを訓練できます：
        1. サンプルデータを使用してモデルを訓練(シミュレーションデータ)
        """)    
       
        if st.button("モデル訓練"):
            with st.spinner("モデルを訓練中..."):
                # シミュレーションデータの生成
                X, y = generate_training_data()

                # モデルの初期化と訓練
                model = VoiceQualityModel()
                model.train(X, y)
                
                # モデルの保存
                st.session_state.ml_model = model
                st.session_state.model_trained = True

                st.success("モデルの訓練が完了しました")
            


    elif page == "本アプリについて":
        st.markdown('<h1 class="sub-header">アプリについて</h1>', unsafe_allow_html=True)
            
        st.write("""
        ## 語尾までしっかりマスター
            
        このアプリは日本語の特性を考慮した音声分析アプリです。特に日本語の文末の音量低下の傾向を検出し、より明確な発話をサポートします。
            
        ### 開発背景
            
        日本語はSOV型の言語であり、文末に述語や重要な情報が集中します。しかし発話中は時間の経過とともに肺の空気が減少し、文末では自然と声量が低下します。特に家族や友人との親密な会話では気が緩み、この傾向が強くなります。
            
               
        ### 日本語のSOV構造と音量低下
            
        日本語のようなSOV構造（Subject-Object-Verb、主語-目的語-動詞）の言語では、文末に動詞や重要な情報が来ることが多いです。例えば：
            
        - 「私は**リンゴを食べます**」（日本語：SOV）
        - "I eat an apple"（英語：SVO）
            
        英語では目的語（apple）が文末にありますが、日本語では動詞（食べます）が文末に来ます。このため、日本語では文末の明瞭さがより重要になります。
            
        ### アプリの機能
            
        - 音声波形と音量変化の可視化
        - 文末音量低下の検出と評価
        - リアルタイムフィードバックの提供
        - 詳細な音声特徴量の分析
        - 練習のためのサンプル文の提供
            
        ### 使用技術
            
        - Python
        - Streamlit
        - librosa（音声処理）
        - WebRTC（リアルタイム音声処理）
        - 機械学習アルゴリズム（特徴量分析）
            
        ### 参考文献
            
        - Chasin M. Sentence final hearing aid gain requirements of some non-English languages. Can J Speech-Lang Pathol Audiol. 2012;36(3):196-203.
        - 日本語日常会話コーパスから見える会話場面と声の高さの関係性 (https://repository.ninjal.ac.jp/records/3193)
        - 小声とは何か、または言語の違いが小声にどのような影響を与えるのか (https://www.oticon.co.jp/hearing-aid-users/blog/2020/20200512)
        """)
            
        st.write("### 留意事項")
        st.info("""
        - 本アプリは、音声データを分析するため、プライバシーに配慮してください
        - 音声データは一時的に保存され、分析後に削除されます
        - 本アプリは、一般的な音声分析を目的としており、特定の個人や状況に対する評価を行うものではありません
        - 本アプリは、専門的な音声分析ツールではなく、あくまで参考としてご利用ください
        """)

# アプリを終了する前に一時ファイルをクリーンアップ
def cleanup():
    if st.session_state.get('temp_audio_file') and os.path.exists(st.session_state.temp_audio_file):
        try:
            os.unlink(st.session_state.temp_audio_file)
        except Exception as e:
            logger.warning(f"一時ファイルの削除に失敗: {e}")

# アプリの実行
if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
