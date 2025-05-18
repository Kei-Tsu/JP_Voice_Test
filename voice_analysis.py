# ml_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

import librosa

def analyze_volume(y, sr):
    """基本的な音量分析を行う関数（後方互換性のため）"""
    extractor = VoiceFeatureExtractor()
    features = extractor.extract_features(y, sr)
    return features 

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
        advice = "語尾までしっかり発話できています！バランスがよい発話です。"
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

# ドロップ率に応じたフィードバックを生成
def get_feedback(drop_rate):
    if drop_rate < 0.1:
        return {
            "level": "good",
            "message": "良い感じです！語尾までしっかり発音できています。",
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
    
def generate_training_data():
    """機械学習用のシミュレーションデータを生成する関数"""
    # サンプルデータの生成（実際には音声データから特徴量を抽出して使用）
    X = []
    y = []
    
    # 「良好」な音声（文末音量低下が小さい）のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.1, 0.3),     # mean_volume
            np.random.uniform(0.02, 0.05),   # std_volume
            np.random.uniform(0.1, 0.3),     # start_volume
            np.random.uniform(0.1, 0.3),     # middle_volume
            np.random.uniform(0.09, 0.25),   # end_volume（あまり低下しない）
            np.random.uniform(0.05, 0.15),   # end_drop_rate（小さい）
            np.random.uniform(0.09, 0.25),   # last_20_percent_volume
            np.random.uniform(0.05, 0.15),   # last_20_percent_drop_rate
            np.random.uniform(1000, 2000),   # spectral_centroid_mean
            np.random.uniform(2, 4),         # speech_rate
        ]

        X.append(features)
        y.append("良好")  # クラスラベル
    
    # 「文末が弱い」音声のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.1, 0.3),     # mean_volume
            np.random.uniform(0.02, 0.05),   # std_volume
            np.random.uniform(0.1, 0.3),     # start_volume
            np.random.uniform(0.1, 0.3),     # middle_volume
            np.random.uniform(0.02, 0.08),   # end_volume（大きく低下）
            np.random.uniform(0.3, 0.5),     # end_drop_rate（大きい）
            np.random.uniform(0.02, 0.08),   # last_20_percent_volume
            np.random.uniform(0.3, 0.5),     # last_20_percent_drop_rate
            np.random.uniform(1000, 2000),   # spectral_centroid_mean
            np.random.uniform(2, 4),         # speech_rate
        ]

        X.append(features)
        y.append("文末が弱い")  # クラスラベル

    # 「小声すぎる」音声のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.01, 0.05),   # mean_volume（全体的に小さい）
            np.random.uniform(0.01, 0.02),   # std_volume
            np.random.uniform(0.01, 0.05),   # start_volume
            np.random.uniform(0.01, 0.05),   # middle_volume
            np.random.uniform(0.01, 0.03),   # end_volume
            np.random.uniform(0.1, 0.3),     # end_drop_rate
            np.random.uniform(0.01, 0.03),   # last_20_percent_volume
            np.random.uniform(0.1, 0.3),     # last_20_percent_drop_rate
            np.random.uniform(800, 1500),    # spectral_centroid_mean
            np.random.uniform(1, 3),         # speech_rate
        ]
        
        X.append(features)
        y.append("小声すぎる")  # クラスラベル
    
    return np.array(X), np.array(y)  


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
        
        # より詳細な文末分析（最後の20%部分）
        end_portion = max(1, int(len(rms) * 0.2))  # 最後の20%
        features['last_20_percent_volume'] = np.mean(rms[-end_portion:])
        features['last_20_percent_drop_rate'] = (features['mean_volume'] - features['last_20_percent_volume']) / features['mean_volume'] if features['mean_volume'] > 0 else 0
        
        # MFCC特徴量（音声の音色特性）
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(len(mfccs)):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # スペクトル特性
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # 音声のペース（オンセット検出で音節を近似）
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        features['onset_count'] = len(onsets)
        features['speech_rate'] = len(onsets) / (len(audio_data) / sr) if len(audio_data) > 0 else 0
        
        return features

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
        
        # より詳細な文末分析（最後の20%部分）
        end_portion = max(1, int(len(rms) * 0.2))  # 最後の20%
        features['last_20_percent_volume'] = np.mean(rms[-end_portion:])
        features['last_20_percent_drop_rate'] = (features['mean_volume'] - features['last_20_percent_volume']) / features['mean_volume'] if features['mean_volume'] > 0 else 0
        
        # MFCC特徴量（音声の音色特性）
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(len(mfccs)):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # スペクトル特性
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # 音声のペース（オンセット検出で音節を近似）
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        features['onset_count'] = len(onsets)
        features['speech_rate'] = len(onsets) / (len(audio_data) / sr) if len(audio_data) > 0 else 0
        
        return features

class VoiceQualityModel:
    """音声品質を評価する機械学習モデル"""

    def __init__(self):
        """モデルの初期化"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = VoiceFeatureExtractor()
        self.is_trained = False
        self.classes = None

    def prepare_features(self, features_dict):
        """特徴辞書から機械学習用の特徴量配列を作成"""
        # 主要な特徴量のみを抽出
        feature_keys = [
            'mean_volume', 'std_volume', 
            'start_volume', 'middle_volume', 'end_volume',
            'end_drop_rate', 'last_20_percent_volume', 'last_20_percent_drop_rate'
        ]

        # 特徴ベクトルの作成
        features = []
        for key in feature_keys:
            if key in features_dict:
                features.append(features_dict[key])
            else:
                features.append(0)  # 特徴が存在しない場合は0で埋める
        
        # MFCCなどの追加特徴（あれば追加）
        if 'spectral_centroid_mean' in features_dict:
            features.append(features_dict['spectral_centroid_mean'])
        if 'speech_rate' in features_dict:
            features.append(features_dict['speech_rate'])

        return features

    def train(self, X, y):
        """モデルを訓練する"""
        # データの標準化（特徴量のスケーリング）
        X_scaled = self.scaler.fit_transform(X)

        # ランダムフォレスト分類器を作成
        self.model = RandomForestClassifier(
            n_estimators=100,  # 決定木の数
            max_depth=10,      # 木の最大深さ
            random_state=42    # 再現性のための乱数シード
        )

        # モデルの訓練
        self.model.fit(X_scaled, y)
        
        # クラスのリストを保存
        self.classes = self.model.classes_
        
        # 訓練済みフラグを設定
        self.is_trained = True

        return True
    
    def predict(self, features_dict):
        """音声品質を予測する"""

        if not self.is_trained or self.model is None:
            return None, 0
    
        # 特徴量配列を作成
        features = self.prepare_features(features_dict)
    
        # 特徴量を2次元配列に変換（sklearn要件）
        features_2d = np.array([features])
    
        # 特徴量を標準化
        features_scaled = self.scaler.transform(features_2d)
    
        # 予測実行
        prediction = self.model.predict(features_scaled)[0]
    
        # 予測確率（信頼度）を取得
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
    
        return prediction, confidence

    def save_model(self, file_path):
        """モデルを保存する"""
        if not self.is_trained or self.model is None:
            # モデルが訓練されていない場合や存在しない場合は保存しない
            return False
    
        # モデル情報を辞書にまとめる
        model_info = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'classes': self.classes
            }
    
        # ファイルに保存
        joblib.dump(model_info, file_path)
        return True

    def load_model(self, file_path):
        """保存されたモデルを読み込む"""
        try:
            # ファイルからモデル情報を読み込む
            model_info = joblib.load(file_path)
        
            # モデル情報を復元
            self.model = model_info['model']
            self.scaler = model_info['scaler']
            self.is_trained = model_info['is_trained']
            self.classes = model_info['classes']
        
            return True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False
 




