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

import av
import scipy.io.wavfile
import time

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


# 音声特徴量抽出のための関数
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

def plot_audio_analysis(features, audio_data, sr):
    """音声分析の視覚化を行う関数"""
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1つ目のプロット: 波形表示
    librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('Amplitude')
    
    # 2つ目のプロット: 音量変化
    rms = features['rms']
    times = features['times']
    ax2.plot(times, rms, color='blue', label='Volume (RMS)')
    ax2.set_title('Volume Change Over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Volume (RMS)')
    
    # 文末部分（最後の20%）を強調表示
    if len(times) > 0:
        end_portion = max(1, int(len(times) * 0.2))  # 最後の20%
        start_highlight = times[-end_portion]
        end_time = times[-1]
        ax2.axvspan(start_highlight, end_time, color='red', alpha=0.2)
        ax2.text(start_highlight + (end_time - start_highlight)/10, 
               max(rms) * 0.8, 'End Part (last 20%)', color='red')
    
    # 文頭・文中・文末の平均音量を水平線で表示
    ax2.axhline(y=features['start_volume'], color='green', linestyle='--', label='Start Volume Average')
    ax2.axhline(y=features['middle_volume'], color='orange', linestyle='--', label='Middle Volume Average')
    ax2.axhline(y=features['end_volume'], color='red', linestyle='--', label='End Volume Average')
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

# リアルタイム音声処理のコールバック関数
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
            
            # 文末検出のための処理
            silence_threshold = -40  # 無音判定の閾値（dB）
            
            # 音量が閾値より大きい場合、音声あり
            if db > silence_threshold:
                st.session_state.last_sound_time = time.time()
                st.session_state.end_of_sentence_detected = False
            else:
                # 無音状態が一定時間続いた場合、文末と判断
                current_time = time.time()
                silence_duration = (current_time - st.session_state.last_sound_time) * 1000  # ミリ秒に変換
                
                if silence_duration > 500 and not st.session_state.end_of_sentence_detected:  # 0.5秒以上の無音
                    st.session_state.end_of_sentence_detected = True
                    
                    # 文末の音量低下率を計算
                    if len(st.session_state.volume_history) > 10:
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
                                "feedback": feedback
                            })
    
    except Exception as e:
        print(f"音声フレーム処理エラー: {e}")
    
    return frame

# ドロップ率に応じたフィードバックを生成
def get_feedback(drop_rate):
    if drop_rate < 0.1:
        return {
            "level": "good",
            "message": "素晴らしい！語尾までしっかり発音できています。",
            "emoji": "🌟"
        }
    elif drop_rate < 0.25:
        return {
            "level": "medium",
            "message": "語尾がやや弱まっています。もう少し意識しましょう。",
            "emoji": "⚠️"
        }
    else:
        return {
            "level": "bad",
            "message": "語尾の音量が大きく低下しています。文末を意識して！",
            "emoji": "❗"
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

# アプリのメイン部分
def main():
    # 特徴抽出器の初期化
    feature_extractor = VoiceFeatureExtractor()
    
    # アプリのタイトルと説明
    st.title('語尾までしっかりマスター')
    st.write('身近な会話をしっかり伝えることで、大切な人とのコミュニケーションを高めよう')

    # サイドバーでナビゲーション
    page = st.sidebar.selectbox("ページ選択", ["ホーム", "練習を始める", "本アプリについて"])
    st.session_state.page = page  # ページ状態を更新

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
                        st.write("素晴らしいです！語尾まで発話できています。")
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
                        
                    # 一時ファイルを削除
                    os.remove(tmp_file_path)
                    st.success("分析が完了しました！")
                        
                except Exception as e:
                    error_msg =str(e)
                    if "PySoundFile" in error_msg:
                        st.error("音声ファイルの形式が正しくありません。別のwavまたはmp3形式のファイルをお試しください。")
                    elif "empty_file" in error_msg:
                        st.error("アップロードがいるされた音声ファイルが空です。有効な音声ファイルをアップロードしてください。")
                    else:
                        st.error(f"音声分析中にエラーが発生しました: {e}")
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

        elif practice_method == "リアルタイム評価":
            st.write("### リアルタイム評価")
            st.info("「START」ボタンをクリックし、ブラウザからのマイク使用許可リクエストを承認してください。その後、サンプル文を読み上げると、リアルタイムで評価が表示されます。")
                    
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
                
                # WebRTC接続が有効な場合
                if webrtc_ctx.state.playing:
                    # 音量メーターの表示
                    display_volume_meter(volume_placeholder)
                            
                    # 状態表示
                    if st.session_state.end_of_sentence_detected:
                        drop_rate = st.session_state.current_drop_rate
                                
                        if drop_rate < 0.1:
                            status_placeholder.success("- 素晴らしいです！語尾までしっかり発音できています。")
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
                        st.write("- -20dB以上: かなり大きな声")
                        st.write("- -30dB～-20dB: 通常の会話音量")
                        st.write("- -40dB～-30dB: 小声")
                        st.write("- -40dB以下: 無音または非常に小さい音")
                else:
                    status_placeholder.warning("マイク接続待機中...「START」ボタンをクリックしてください。")
            
            except Exception as e:
                st.error(f"マイク機能でエラーが発生しました: {e}")
                st.info("お使いのブラウザがWebRTCに対応していないか、マイクへのアクセス許可がない可能性があります。")

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

# アプリの実行
if __name__ == "__main__":
    main()
    print("アプリが起動されました")