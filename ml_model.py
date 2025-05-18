# ml_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import librosa

# 必要なクラスと関数を実装
class VoiceQualityModel:
    """音声品質を評価する機械学習モデル"""

    def __init__(self):
        """モデルの初期化"""
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = None

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
    
    def predict(self, features):
        """音声品質を予測する"""
        if not self.is_trained or self.model is None:
            return None, 0
    
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

# シミュレーションデータ生成関数
def generate_training_data():
    """機械学習用のシミュレーションデータを生成する関数"""
    # サンプルデータの生成
    X = []
    y = []
    
    # 「良好」な音声のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.1, 0.3),     # mean_volume
            np.random.uniform(0.02, 0.05),   # std_volume
            np.random.uniform(0.1, 0.3),     # start_volume
            np.random.uniform(0.1, 0.3),     # middle_volume
            np.random.uniform(0.09, 0.25),   # end_volume
            np.random.uniform(0.05, 0.15),   # end_drop_rate
            np.random.uniform(0.09, 0.25),   # last_20_percent_volume
            np.random.uniform(0.05, 0.15),   # last_20_percent_drop_rate
            np.random.uniform(1000, 2000),   # spectral_centroid_mean
            np.random.uniform(2, 4),         # speech_rate
        ]
        X.append(features)
        y.append("良好")
    
    # 「文末が弱い」音声のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.1, 0.3),     # mean_volume
            np.random.uniform(0.02, 0.05),   # std_volume
            np.random.uniform(0.1, 0.3),     # start_volume
            np.random.uniform(0.1, 0.3),     # middle_volume
            np.random.uniform(0.02, 0.08),   # end_volume
            np.random.uniform(0.3, 0.5),     # end_drop_rate
            np.random.uniform(0.02, 0.08),   # last_20_percent_volume
            np.random.uniform(0.3, 0.5),     # last_20_percent_drop_rate
            np.random.uniform(1000, 2000),   # spectral_centroid_mean
            np.random.uniform(2, 4),         # speech_rate
        ]
        X.append(features)
        y.append("文末が弱い")
    
    # 「小声すぎる」音声のデータ
    for _ in range(50):
        features = [
            np.random.uniform(0.01, 0.05),   # mean_volume
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
        y.append("小声すぎる")
    
    return np.array(X), np.array(y)