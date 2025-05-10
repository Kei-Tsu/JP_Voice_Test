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
import matplotlib.pyplot as plt # グラフ描画用

# WebRTC関連のライブラリのインポート
from streamlit_webrtc import webrtc_streamer, WebRtcMode # ブラウザで音声を録音するためのライブラリ
from streamlit_webrtc import RTCConfiguration

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

import av
import scipy.io.wavfile
import time

def get_file_extension(filename):
    """ファイル名から拡張子を安全に取得する関数"""
    filename = filename.lower()
    if filename.endswith(".wav"):
        return ".wav"
    elif filename.endswith(".mp3"):
        return ".mp3"
    else:
        return None

def is_audio_file(filename):
    """オーディオファイルかどうかを判定する関数"""
    ext = get_file_extension(filename)
    return ext in [".wav", ".mp3"]

# 変数を初期化
uploaded_file = None  # 変数を事前に定義する

# セッション状態の初期化
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = {"is_recording": False} # 録音中かどうかのフラグ
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = [] # 録音したフレームを保存するリスト
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio_path = None # 録音した音声ファイルのパス
if 'volume_history' not in st.session_state:
    st.session_state.volume_history = []  # 音量履歴（グラフ表示用）
if 'last_sound_time' not in st.session_state:
    st.session_state.last_sound_time = time.time()  # 最後に音が検出された時間
if 'silence_threshold' not in st.session_state:
    st.session_state.silence_threshold = -35  # 無音と判定するdBのしきい値
if 'auto_stop_duration' not in st.session_state:
    st.session_state.auto_stop_duration = 1000  # 自動停止する無音時間（ミリ秒）
if "input_method" not in st.session_state:
    st.session_state.input_method = "録音する"  # デフォルトの入力方法  

def start_recording():
    """録音を開始する関数"""
    st.session_state.recording_state["is_recording"] = True
    st.session_state.recorded_frames = []  # 録音フレームをクリア
    st.toast("録音を開始しました", icon="🎤")
    
def stop_recording():
    """録音を停止する関数"""
    st.session_state.recording_state["is_recording"] = False
    st.toast("録音を停止しました", icon="✅")
    # 録音フレームがある場合は処理
    if len(st.session_state.recorded_frames) > 0:
        return True
    return False

# グローバル変数（WebRTCコールバック用）
global_recording_state = {"is_recording": False}
global_recorded_frames = []

# リアルタイム音声処理用のキュー
audio_queue = queue.Queue()
# recorded_frames = []
# recording_state = {"is_recording": False}

# 音声フレーム処理のコールバック関数
def audio_frame_callback(frame):
    """音声フレームを処理するコールバック関数"""
    
    try:
        sound = frame.to_ndarray()

        # グローバル録音状態の確認
        if global_recording_state["is_recording"]:
            # 録音中の場合のみフレームを保存
            global_recorded_frames.append(sound.copy())
            
    except Exception as e:
        print(f"フレーム処理エラー: {e}")
    
    return frame
          
    # 現在の音量レベルを計算
    audio_data = sound.flatten() 
    if len(audio_data) > 0:
        # RMS(二乗平均平方根）で音量を計算
        rms = np.sqrt(np.mean(audio_data**2))  # RMS音量計算
        db = 20 * np.log10(max(rms, 1e-10))  # dBに変換(非常に小さい値の場合の対策)

        # 音量履歴に追加（グラフ表示用）
        if 'volume_history' in st.session_state:
            st.session_state.volume_history.append({"音量": db})
            # 音量履歴の長さを制限(パフォーマンス向上のため)
                     

        # 録音中の場合のみフレームを保存（メモリ使用量を減らすため）
        if st.session_state.recording_state ["is_recording"]:
            st.session_state.recorded_frames.append(sound.copy())  # 録音フレームを保存

def audio_frame_callback(frame):
    """音声フレームを処理するコールバック関数"""
    try:
        sound = frame.to_ndarray()
    except Exception as e:
        st.error(f"Error processing audio frame: {e}")
        return frame

def show_debug_info(webrtc_ctx):
    """デバッグ情報を表示する関数"""   
    # デバッグ情報の表示オン/オフを切り替えるためのチェックボックス
    if st.checkbox("デバッグ情報を表示", value=False):
        st.write("### デバッグ情報")
        st.write(f"WebRTCの状態: {webrtc_ctx.state}")
        st.write(f"録音状態: {st.session_state.recording_state}")
        st.write(f"録音フレーム数: {len(st.session_state.recorded_frames)}")

        # 追加のデバッグ情報
        if 'current_silence_duration' in st.session_state:
            st.write(f"現在の無音時間: {st.session_state.current_silence_duration:.2f} ms")

        if 'volume_history' in st.session_state:
            st.write(f"音量履歴データ数: {len(st.session_state.volume_history)}")

        # 最新の音量を表示
        if 'volume_history' in st.session_state and len(st.session_state.volume_history) > 0:
            st.write(f"最新の音量履歴: {st.session_state.volume_history[-1]['音量']:.2f} dB")
            
# 音量メーター表示用の関数
def display_volume_meter(placeholder):
    """リアルタイム音量メーターを表示する関数"""
    if 'volume_history' in st.session_state and len(st.session_state.volume_history) > 0:
        # 音量履歴をデータフレームに変換
        df = pd.DataFrame(st.session_state.volume_history)
        df = df.reset_index().rename(columns={"index": "時間"})

        # Altairを使ったグラフ表示
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("時間", axis=None),  # x軸ラベルを非表示
            y=alt.Y("音量", title="音量 (dB)", scale=alt.Scale(domain=[-80, 0]))
        ).properties(
            height=200,
            width='container'
        )

        # プレースホルダーにグラフを表示
        placeholder.altair_chart(chart, use_container_width=True)

def recording_controls(webrtc_ctx, status_placeholder, recording_status_placeholder):
    """録音操作のUI部分"""
    if webrtc_ctx.state.playing:
        # WebRTC接続が有効な場合
        status_placeholder.success("マイクが接続されています", icon="✅")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 録音開始ボタン（録音中は無効化）
            start_button = st.button(
                "録音開始", 
                disabled=st.session_state.recording_state["is_recording"],
                key="start_rec_button"
            )
            if start_button:
                # 録音開始処理
                st.session_state.recording_state["is_recording"] = True
                st.session_state.recorded_frames = []  # 録音データをクリア
                recording_status_placeholder.warning("録音中... 話し終わったら「録音停止」ボタンを押してください。", icon="🎙️")
                st.toast("録音を開始しました", icon="🎙️")
                st.experimental_rerun()  # UIを更新
            with col2:
                # 録音停止ボタン（録音中のみ有効）
                stop_button = st.button(
                    "録音停止", 
                    disabled=not st.session_state.recording_state["is_recording"],
                    key="stop_rec_button"
                )
                if stop_button:
                    # 録音停止処理
                    st.session_state.recording_state["is_recording"] = False
                    recording_status_placeholder.success("録音が停止されました。", icon="✅")
                    st.toast("録音を停止しました", icon="✅")

                    # 録音データの処理（フレームの結合と保存）
                    if len(st.session_state.recorded_frames) > 0:
                        # ここで process_recorded_audio() を呼び出す予定
                        recording_status_placeholder.success("録音データの準備ができました", icon="✅")
                    else:
                        recording_status_placeholder.error("録音データがありません", icon="❌")
                    
                    st.experimental_rerun()  # UIを更新

    else:
        # WebRTC接続が無効な場合
        status_placeholder.warning("マイクが接続されていません。「START」ボタンを押してくマイクを有効化してください。", icon="🎤")

def configure_webrtc():
    """WebRTCの設定を行う関数"""
    return webrtc_streamer(
        key="speech-recorder-config",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ],
            "iceTransportPolicy": "all",
            "iceCandidatePoolSize": 10,
            "bundlePolicy": "max-bundle",
            "rtcpMuxPolicy": "require"
        },
        media_stream_constraints={
            "video": False, 
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        async_processing=True,
    )
def process_recorded_audio(analysis_placeholder):
    """録音データを処理して音声分析を行う関数"""
    if len(st.session_state.recorded_frames) == 0:
        st.warning("録音データがありません。再度録音してください。")
        return
    
    try:
        # 録音フレームを結合
        audio_data = np.concatenate(st.session_state.recorded_frames, axis=0)
        sample_rate = 48000  # WebRTCのデフォルトサンプルレート
        
        # WAVファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, sample_rate, audio_data)
            st.session_state.recorded_audio_path = tmp_file.name
            
        # 音声分析
        y, sr = librosa.load(st.session_state.recorded_audio_path, sr=None)
        
        # 特徴量抽出（既存の関数を利用）
        features = feature_extractor.extract_features(y, sr)
        
        # 分析結果表示用のコンテナを作成
        with analysis_placeholder.container():
            st.subheader("録音された音声")
            st.audio(st.session_state.recorded_audio_path, format="audio/wav")
            
            # 音声波形と音量変化のグラフ表示
            st.subheader("音声分析")
            fig = plot_audio_analysis(features, y, sr)
            st.pyplot(fig)
            
            # 音量分析結果のメトリクス表示
            st.markdown('<h2 class="sub-header">音量分析結果</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
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
                st.metric("文末音量低下率", f"{features['end_drop_rate']:.2f}")
                if 'last_20_percent_drop_rate' in features:
                    st.metric("最後の20%音量低下率", f"{features['last_20_percent_drop_rate']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 明瞭度評価
            evaluation = evaluate_clarity(features)
            
            st.markdown('<h2 class="sub-header">明瞭度評価</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("明瞭度スコア", f"{evaluation['score']}/100")
                st.metric("評価", evaluation["clarity_level"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("アドバイス")
                st.write(evaluation["advice"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 追加: 詳細な特徴量表示（オプション）
            if st.checkbox("詳細な特徴量を表示", key="show_detailed_features"):
                st.subheader("抽出された特徴量")
                
                # 特徴量をカテゴリ別に整理して表示
                volume_features = {k: v for k, v in features.items() if 'volume' in k or 'rms' in k or 'drop' in k}
                spectral_features = {k: v for k, v in features.items() if 'spectral' in k or 'mfcc' in k}
                rhythm_features = {k: v for k, v in features.items() if 'onset' in k or 'speech' in k}
                
                st.write("### 音量関連特徴量")
                st.write(pd.DataFrame({
                    '特徴量': list(volume_features.keys()),
                    '値': list(volume_features.values())
                }))
                
                if spectral_features:
                    st.write("### スペクトル特徴量")
                    st.write(pd.DataFrame({
                        '特徴量': list(spectral_features.keys()),
                        '値': list(spectral_features.values())
                    }))
                
                if rhythm_features:
                    st.write("### リズム関連特徴量")
                    st.write(pd.DataFrame({
                        '特徴量': list(rhythm_features.keys()),
                        '値': list(rhythm_features.values())
                    }))
        
        # 一時ファイルのクリーンアップは必要に応じて実装
        
    except Exception as e:
        # エラーハンドリング
        st.error(f"音声処理中にエラーが発生しました: {e}")
        # スタックトレースも表示（デバッグ用）
        import traceback
        st.code(traceback.format_exc())
                         

# 会話サンプル
CONVERSATION_SAMPLES = {
    "家族との会話": [
        "今日の夕食、何にする？パスタでいい？",
        "さっきのニュース見た？なんか面白かったね",
        "土曜日は何か予定ある？買い物に行こうかなと思ってるんだけど"
    ],
    "友人との会話": [
        "この間の話の続きなんだけど、結局どうなったの？",
        "新しいカフェ見つけたんだ。今度一緒に行かない？",
        "最近どう？何か変わったことあった？"
    ],
    "恋人との会話": [
        "ちょっとこれ手伝ってもらえる？すぐ終わるから",
        "窓開けてもらってもいい？ちょっと暑いと思って",
        "あのね、昨日見た映画がすごく良かったんだ"
    ]
}
# 音声分析のためのクラスと関数の定義
class VoiceFeatureExtractor:
    """音声から特徴量を抽出するクラス"""
    
    def extract_features(self, audio_data, sr):
        """音声特徴量を抽出する関数"""
        features = {}
        
        # 基本的な音量特徴量（RMS）
        rms = librosa.feature.rms(y=audio_data)[0]
        times = librosa.times_like(rms, sr=sr)
        features['rms'] = rms
        features['times'] = times
        features['mean_volume'] = np.mean(rms)
        features['std_volume'] = np.std(rms)
        features['max_volume'] = np.max(rms)
        features['min_volume'] = np.min(rms)
        
        # 会話音声を三分割した分析
        third = len(rms) // 3
        features['start_volume'] = np.mean(rms[:third])  # 最初の1/3
        features['middle_volume'] = np.mean(rms[third:2*third])  # 中間の1/3
        features['end_volume'] = np.mean(rms[2*third:])  # 最後の1/3
        
        # 文末音量低下率の計算
        features['end_drop_rate'] = (features['middle_volume'] - features['end_volume']) / features['middle_volume'] if features['middle_volume'] > 0 else 0
        
        # 追加: より詳細な文末分析（最後の20%部分）
        end_portion = max(1, int(len(rms) * 0.2))  # 最後の20%
        features['last_20_percent_volume'] = np.mean(rms[-end_portion:])
        features['last_20_percent_drop_rate'] = (features['mean_volume'] - features['last_20_percent_volume']) / features['mean_volume'] if features['mean_volume'] > 0 else 0
        
        # 追加: MFCC特徴量（音声の音色特性）
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(len(mfccs)):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # 追加: スペクトル特性
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # 追加: 音声のペース（オンセット検出で音節を近似）
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        features['onset_count'] = len(onsets)
        features['speech_rate'] = len(onsets) / (len(audio_data) / sr) if len(audio_data) > 0 else 0
        
        return features

def analyze_volume(y, sr):
    """基本的な音量分析を行う関数（後方互換性のため）"""
    extractor = VoiceFeatureExtractor()
    features = extractor.extract_features(y, sr)
    
    # 元の関数と同じキーと値を返す
    return {
        "rms": features["rms"],
        "times": features["times"],
        "mean_volume": features["mean_volume"],
        "std_volume": features["std_volume"],
        "max_volume": features["max_volume"],
        "min_volume": features["min_volume"],
        "start_volume": features["start_volume"],
        "middle_volume": features["middle_volume"],
        "end_volume": features["end_volume"],
        "end_drop_rate": features["end_drop_rate"]
    }

def plot_audio_analysis(features, audio_data, sr):
    """音声分析の視覚化を行う関数"""
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1つ目のプロット: 波形表示
    librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
    ax1.set_title('音声波形')
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('振幅')
    
    # 2つ目のプロット: 音量変化
    rms = features['rms']
    times = features['times']
    ax2.plot(times, rms, color='blue', label='音量 (RMS)')
    ax2.set_title('音量変化')
    ax2.set_xlabel('時間 (秒)')
    ax2.set_ylabel('音量 (RMS)')
    
    # 文末部分（最後の20%）を強調表示
    if len(times) > 0:
        end_portion = max(1, int(len(times) * 0.2))  # 最後の20%
        start_highlight = times[-end_portion]
        end_time = times[-1]
        ax2.axvspan(start_highlight, end_time, color='red', alpha=0.2)
        ax2.text(start_highlight + (end_time - start_highlight)/10, 
               max(rms) * 0.8, '文末部分 (最後の20%)', color='red')
    
    # 文頭・文中・文末の平均音量を水平線で表示
    ax2.axhline(y=features['start_volume'], color='green', linestyle='--', label='文頭平均')
    ax2.axhline(y=features['middle_volume'], color='orange', linestyle='--', label='文中平均')
    ax2.axhline(y=features['end_volume'], color='red', linestyle='--', label='文末平均')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def evaluate_clarity(features):
    """音量特徴からクリアな発話かどうかを評価する関数"""
    drop_rate = features["end_drop_rate"]
    last_20_drop_rate = features.get("last_20_percent_drop_rate", 0)  # キーがない場合は0
    
    # 両方のドロップ率を考慮した評価
    avg_drop_rate = (drop_rate + last_20_drop_rate) / 2
    
    if avg_drop_rate < 0.1:
        clarity_level = "良好"
        advice = "語尾までしっかり発話できています！素晴らしいバランスです。"
        score = min(100, int((1 - avg_drop_rate) * 100))
    elif avg_drop_rate < 0.25:
        clarity_level = "普通"
        advice = "語尾がやや弱まっています。もう少し文末を意識すると良いでしょう。"
        score = int(75 - (avg_drop_rate - 0.1) * 100)
    elif avg_drop_rate < 0.4:
        clarity_level = "やや弱い"
        advice = "文末の音量がかなり低下しています。文末を1音上げるイメージで話してみましょう。"
        score = int(60 - (avg_drop_rate - 0.25) * 100)
    else:
        clarity_level = "改善必要"
        advice = "語尾の音量が大きく低下しています。日本語は文末に重要な情報や結論が来ることが多いです、文末まで意識して相手に伝える練習をしましょう。"
        score = max(20, int(40 - (avg_drop_rate - 0.4) * 50))
    
    return {
        "clarity_level": clarity_level,
        "advice": advice,
        "score": score,
        "avg_drop_rate": avg_drop_rate
    }

# インスタンスを作成
feature_extractor = VoiceFeatureExtractor()

# アプリのタイトルと説明
st.title('語尾までしっかりマスター')
st.write('身近な会話をしっかり伝えることで、大切な人とのコミュニケーションを高めよう')

# サイドバーでナビゲーション
page = st.sidebar.selectbox("ページ選択", ["ホーム", "練習を始める", "本アプリについて"])

# 特徴抽出器の初期化
feature_extractor = VoiceFeatureExtractor()

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
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">使い方</h2>', unsafe_allow_html=True)
    st.write("1. 下のボタン、または左のサイドバーから「練習を始める」を選択")
    st.write("2. 練習したいサンプル文を選んで読み上げる")
    st.write("3. 音声ファイルをアップロード")
    st.write("4. 分析結果とアドバイスを確認")
    
    if st.button("練習を始める"):
        st.session_state.page = "練習を始める"
        st.experimental_rerun() 


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

    # 音声入力方法の選択
    st.session_state.input_method = st.radio("音声入力方法", ["録音する", "ファイルをアップロード"],key="unique_key_1") 

    # プレースホルダーの準備（動的更新用）
    status_placeholder = st.empty()
    volume_placeholder = st.empty()
    recording_status_placeholder = st.empty()
    analysis_placeholder = st.empty()
    

    # セッション状態にデフォルト値を設定
    if "input_method" not in st.session_state:
        st.session_state.input_method = "録音する"
  
    #入力方法に応じた処理
    if st.session_state.input_method == "録音する":
        st.write("録音機能が選択されました")
        st.info("「START」ボタンをクリックし、ブラウザからのマイク使用許可リクエストを承認してください。")
    
    elif st.session_state.input_method == "ファイルをアップロード":
        uploaded_file = st.file_uploader(
            "音声ファイルをアップロードしてください", 
            type=["wav", "mp3"],
            key="simple_uploader"
            )
        if uploaded_file is not None:
            st.write(f"アップロードされたファイル: {uploaded_file.name}")
    
    # 音声入力方法の選択（ラジオボタン） 
    st.session_state.input_method = st.radio("音声入力方法", ["録音する", "ファイルをアップロード"], key="input_method_radio_unique")
          
    # 録音する場合の処理
    if st.session_state.input_method == "録音する":
        st.write("### マイクで録音")
        st.info("「START」ボタンをクリックし、ブラウザからのマイク使用許可リクエストを承認してください。")  

    # 録音コントロール表示
    st.session_state.input_method = st.radio("音声入力方法", ["録音する", "ファイルをアップロード"], key="input_method_radio_2")
    
    webrtc_ctx = configure_webrtc() 

    # WebRTC接続が有効な場合の処理
    if webrtc_ctx.state.playing:
        # リアルタイム音量メーター表示
        display_volume_meter(volume_placeholder)

    # デバッグ情報表示
        show_debug_info(webrtc_ctx)
  
    # WebRTC音声ストリーミング    

    webrtc_ctx = webrtc_streamer(
        key="speech-recorder-main",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,  
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                ],
            "iceTransportPolicy": "all",
            "iceCandidatePoolSize": 10,
            "bundlePolicy": "max-bundle",
            "rtcpMuxPolicy": "require"
        },
        media_stream_constraints={
            "video": False, 
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        async_processing=False, # 非同期処理を無効にする
    )

    # WebRTC接続が有効な場合
    if webrtc_ctx.state.playing:
        st.success("WebRTC接続が確立されました。録音を開始します。")
        
        # 録音ボタンを表示
        col1, col2 = st.columns(2)

        with col1:
            if st.button("録音開始", key="start_recording"):

                st.session_state.recording_state["is_recording"] = True
                st.session_state.recorded_frames = []      
                # グローバル変数を更新
                global_recording_state["is_recording"] = True
                global_recorded_frames.clear()  # 録音フレームをクリア
                st.success("録音を開始しました。サンプル文を読み上げてください。")
                st.write(f"録音中フラグ（session）: {st.session_state.recording_state['is_recording']}")
                st.write(f"録音中フラグ（global）: {global_recording_state['is_recording']}")
        
        with col2:
            if st.button("録音停止", key="stop_rec1"):
                st.write("録音停止ボタンが押されました")
                st.write(f"録音フレーム数（global）: {len(global_recorded_frames)}")

                # セッション状態を更新
                st.session_state.recording_state["is_recording"] = False

                # グローバル変数を更新
                global_recording_state["is_recording"] = False
                
                # グローバル変数から録音フレームを取得する
                if len(global_recorded_frames) > 0:
                    try:
                        # フレームを結合
                        audio_data = np.concatenate(global_recorded_frames, axis=0)
                        sample_rate = 48000  # WebRTCのデフォルトサンプルレート               
                
                        # WAVファイルとして保存
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            scipy.io.wavfile.write(tmp_file.name, sample_rate, audio_data)
                            st.session_state.recorded_audio = tmp_file.name
                            #session_stateに録音ファイルのパスを保存(UIの更新用)
                            st.session_state.recorded_frames = global_recorded_frames.copy()  # ← ここ追加！
                            st.success(f"録音完了！ファイル：{tmp_file.name}")
                           
                    except Exception as e:
                        st.error(f"録音ファイルの作成中にエラーが発生しました: {e}")
            else:
                st.warning("WebRTC接続: 無効。「START」ボタンを押してから音声を録音してください。")
        

    #録音状態のデバッグ情報を表示
    st.write("## デバッグ情報")
    st.write(f"WebRTCの状態: {webrtc_ctx.state}")
    st.write(f"録音状態: {global_recording_state}")
    st.write(f"録音フレーム数: {len(global_recorded_frames)}")
    st.write(f"audio_queueの型: {type(audio_queue)}")

    if 'recorded_audio' in st.session_state:
        st.write(f"録音ファイルのパス: {st.session_state.recorded_audio}")
    else:
        st.write("録音ファイルはまだありません")
    
    # WebRTCの詳細デバッグ情報を追加
    if st.checkbox("詳細なWebRTCデバッグ情報を表示"):
        st.write("### 詳細WebRTCデバッグ情報")
        st.write(f"WebRTCオブジェクトの属性: {dir(webrtc_ctx)}")
    
    if webrtc_ctx.state.playing:
        st.write("WebRTC接続が確立されました: 有効")
        
        # オーディオトラック情報の取得を試みる
        try:
            st.write("接続状態の詳細確認中...")
            # 進行状況バーを表示
            progress_bar = st.progress(0)
            for i in range(1, 4):
                time.sleep(0.5)  # 少し待機
                progress_bar.progress(i/3)
                st.write(f"状態チェック {i}: {webrtc_ctx.state}")
            
            st.success("WebRTC接続は正常に動作しています")
        except Exception as e:
            st.error(f"WebRTC詳細取得エラー: {e}")
                    # フレームがあるか確認
        if len(st.session_state.recorded_frames) > 0:
                    try:
                        # 録音データの処理（フレームの結合と保存）
                        audio_data = np.concatenate(st.session_state.recorded_frames, axis=0)               
                
                        # WAVファイルとして保存
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            sample_rate = 48000  # WebRTCのデフォルトサンプルレート
                            scipy.io.wavfile.write(tmp_file.name, sample_rate, audio_data)
                            st.session_state.recorded_audio = tmp_file.name
                            st.success(f"録音完了！ファイル：{tmp_file.name}")
                    except Exception as e:
                        st.error(f"録音ファイルの作成中にエラーが発生しました: {e}")
        else:
                #接続が無効な場合
                st.warning("録音フレームがありません。「録音開始」ボタンを押してから音声を録音してください。")    
               
        # デバッグ情報の表示
        st.write("停止ボタン押下後:")
        st.write(f"録音状態: {st.session_state.recording_state}")
        st.write(f"録音フレーム数: {len(st.session_state.recorded_frames)}")
                
        if len(st.session_state.recorded_frames) > 0:
                    # 録音データを一時ファイルに保存
                    audio_data = np.concatenate(st.session_state.recorded_frames, axis=0) 
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        # サンプルレート仮定（実際はwebrtc_ctxから取得するべき）
                        sample_rate = 48000
                        import scipy.io.wavfile as wavfile
                        wavfile.write(tmp_file.name, sample_rate, audio_data)
                        st.session_state.recorded_audio = tmp_file.name
                        st.success("録音完了！以下で再生と分析ができます")

        
        # 録音データがある場合は表示と分析
        if 'recorded_audio' in st.session_state and st.session_state.recorded_audio:
            if os.path.exists(st.session_state.recorded_audio): # 録音ファイルが存在する場合
               st.audio(st.session_state.recorded_audio, format='audio/wav')
               
               if st.button("録音した音声を分析"):
                try:
                    # デバッグ情報の表示
                    st.write("分析ボタン押下:")
                    st.write(f"録音ファイルパス: {st.session_state.recorded_audio}")
                    st.write(f"ファイルの存在確認: {os.path.exists(st.session_state.recorded_audio)}")

                    # 音声データの読み込み
                    y, sr = librosa.load(st.session_state.recorded_audio, sr=None)  
                    
                    # 読み込み後のデバッグ情報
                    st.write(f"音声データの長さ: {len(y)}")
                    st.write(f"サンプルレート: {sr}")
                    
                    # 特徴量抽出
                    features = analyze_volume(y, sr)
                    
                    # 以下、分析表示のコード
                    # 波形の表示
                    st.subheader("音声波形")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    ax.set_xlabel('時間 (秒)')
                    ax.set_ylabel('振幅')
                    st.pyplot(fig)
                    
                    # 音量変化のグラフ表示
                    st.subheader("音量分析")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(features["times"], features["rms"], color='blue')
                    ax.set_xlabel('時間 (秒)')
                    ax.set_ylabel('音量 (RMS)')
                    
                    # 文末部分（最後の1/3）を強調表示
                    if len(features["times"]) > 0:
                        third = len(features["times"]) // 3
                        start_highlight = features["times"][2*third]
                        end_time = features["times"][-1]
                        ax.axvspan(start_highlight, end_time, color='red', alpha=0.2)
                        ax.text(start_highlight + (end_time - start_highlight)/10, 
                               max(features["rms"]) * 0.8, '文末部分', color='red')
                    st.pyplot(fig)
                    
                    # 音量分析結果の表示（現在のコードと同様）
                    st.subheader("音量分析結果")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("平均音量", f"{features['mean_volume']:.4f}")
                        st.metric("文頭音量", f"{features['start_volume']:.4f}")
                    
                    with col2:
                        st.metric("最大音量", f"{features['max_volume']:.4f}")
                        st.metric("文中音量", f"{features['middle_volume']:.4f}")
                    
                    with col3:
                        st.metric("最小音量", f"{features['min_volume']:.4f}")
                        st.metric("文末音量", f"{features['end_volume']:.4f}")
                    
                    # 文末音量低下率と簡単な評価
                    st.subheader("文末音量分析")
                    
                    drop_rate = features["end_drop_rate"]
                    st.metric("文末音量低下率", f"{drop_rate:.2f}")
                    
                    if drop_rate < 0.1:
                        st.success("語尾までしっかり発音できています！")
                    elif drop_rate < 0.3:
                        st.info("語尾がやや弱まっています。もう少し意識してみましょう。")
                    else:
                        st.warning("語尾の音量が大きく低下しています。文末まで意識して発音する練習をしましょう。")
                
                except Exception as e:
                    st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")

# ファイルアップロードの処理
import os

# 正しいファイル拡張子のチェックを強化
def is_valid_audio_file(file):
    if file is None:
        return False
    # ファイル名を小文字に変換して拡張子をチェック
    filename = file.name.lower()
    return filename.endswith(".wav") or filename.endswith(".mp3")

if st.session_state.input_method == "ファイルをアップロード":
    
    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロードしてください", 
        type=["wav", "mp3"],
        key="uploader123"
        )
    
    # アップロードされたファイルが有効かどうかをチェック
    if uploaded_file is not None:
        file_name =uploaded_file.name
            # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
      
    #デバッグ情報
    if uploaded_file is not None:
        st.write("### デバッグ情報")
        st.write(f"アップロードされたファイル名: {uploaded_file.name}")
        st.write(f"ファイルタイプ: {uploaded_file.type}")
        st.write(f"ファイルサイズ: {uploaded_file.size} バイト")

        #ファイル名から拡張子を取得
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        st.write(f"抽出されたファイルの拡張子: {file_ext}")
        st.write(f'許可されている拡張子:[".wav",".mp3"]')
        st.write(f'拡張子は許可リストに含まれています: {file_ext in [".wav", ".mp3"]}')

   
        # 音声ファイルを再生可能に表示
        audio_format = "audio/wav" if file_ext == ".wav" else "audio/mp3"
        st.audio(tmp_file_path, format=audio_format)
                
        try:
            # 音声データの読み込み
            y, sr = librosa.load(tmp_file_path, sr=None)
            
            # 特徴量抽出
            features = feature_extractor.extract_features(y, sr)
            
            # 音声分析のグラフ表示
            st.subheader("音声分析")
            fig = plot_audio_analysis(features, y, sr)
            st.pyplot(fig)
            
            # 音量分析結果を表示
            st.markdown('<h2 class="sub-header">音量分析結果</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
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
                st.metric("文末音量低下率", f"{features['end_drop_rate']:.2f}")
                st.metric("最後の20%音量低下率", f"{features['last_20_percent_drop_rate']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 明瞭度評価
            evaluation = evaluate_clarity(features)
            
            st.markdown('<h2 class="sub-header">明瞭度評価</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("明瞭度スコア", f"{evaluation['score']}/100")
                st.metric("評価", evaluation["clarity_level"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("アドバイス")
                st.write(evaluation["advice"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 追加: 詳細な特徴量表示（オプション）
            if st.checkbox("詳細な特徴量を表示"):
                st.subheader("抽出された特徴量")
                
                # 特徴量をカテゴリ別に整理して表示
                volume_features = {k: v for k, v in features.items() if 'volume' in k or 'rms' in k or 'drop' in k}
                spectral_features = {k: v for k, v in features.items() if 'spectral' in k or 'mfcc' in k}
                rhythm_features = {k: v for k, v in features.items() if 'onset' in k or 'speech' in k}
                
                st.write("### 音量関連特徴量")
                st.write(pd.DataFrame({
                    '特徴量': list(volume_features.keys()),
                    '値': list(volume_features.values())
                }))
                
                st.write("### スペクトル特徴量")
                st.write(pd.DataFrame({
                    '特徴量': list(spectral_features.keys()),
                    '値': list(spectral_features.values())
                }))
                
                st.write("### リズム関連特徴量")
                st.write(pd.DataFrame({
                    '特徴量': list(rhythm_features.keys()),
                    '値': list(rhythm_features.values())
                }))
            
            # 一時ファイルを削除
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")
            try:
                os.unlink(tmp_file_path)
            except:
                pass

elif page == "アプリについて":
    st.header("アプリについて")
    st.write("""
    ## 語尾マスター
    
    このアプリは日本語の特性を考慮した音声分析アプリです。
    特に日本語の文末の音量低下の傾向を検出し、より明確な発話をサポートします。
    
    ### 開発目的
    - 日本語の語尾をはっきり伝えるトレーニングをサポート
    - 特に親密な1対1での会話における明瞭さの向上を目指します
    - 機械学習を活用した音声分析を実現します
    - 本アプリは、音声データを分析し、特に文末の音量低下を検出することで、発話の明瞭さを評価します
    
    ### 留意事項
    - 本アプリは、音声データを分析するため、プライバシーに配慮してください
    - 音声データは一時的に保存され、分析後に削除されます
    - 本アプリは、一般的な音声分析を目的としており、特定の個人や状況に対する評価を行うものではありません
    - 本アプリは、専門的な音声分析ツールではなく、あくまで参考としてご利用ください
             
    
    ### アプリの機能
    - 音声波形の表示と分析
    - 文末音量低下の検出
    - 適切なフィードバックの提供
    """)

# アプリの起動字の確認メッセージ
if __name__ == "__main__":
    print("アプリが起動されました")

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
</style>
""", unsafe_allow_html=True)
