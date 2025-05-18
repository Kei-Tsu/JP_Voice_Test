import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import pandas as pd

# ロガーの設定
logger = logging.getLogger(__name__)

# 必要なクラスと関数を実装
class VoiceQualityModel:
    """音声品質を評価する機械学習モデル"""

    def __init__(self):
        """モデルの初期化"""
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = None
        self.feature_names = None  # 特徴量名を記録

    def prepare_features(self, features_dict):
        """特徴辞書から機械学習用の特徴量配列を作成"""
        # 主要な特徴量のリスト
        feature_keys = [
            'mean_volume', 'std_volume', 
            'start_volume', 'middle_volume', 'end_volume',
            'end_drop_rate', 'last_20_percent_volume', 'last_20_percent_drop_rate',
            'spectral_centroid_mean', 'speech_rate'
        ]

        # 特徴ベクトルの作成
        features = []
        for key in feature_keys:
            if key in features_dict:
                value = features_dict[key]
                # Nanや無限大価の処理（0に置き換え）
                if np.isnan(value) or np.isinf(value):
                    features.append(0.0)
                else:
                    features.append(float(value))   
            else:
                features.append(0.0) # 特徴値が存在しない場合デフォルト値0で埋めるため追加
    
            return features
        
        # 特徴量名を記録（初回のみ適用）
        if self.feature_names is None:
            self.feature_names = feature_keys

        return features        

    def prepare_features_realtime(self, features_dict):
        """リアルタイム用の軽量特徴量配列を作成"""
        # リアルタイム用の基本特徴量のみ
        feature_keys = [
            'mean_volume', 'std_volume', 
            'middle_volume', 'end_volume', 'end_drop_rate'
        ]

        features = []
        for key in feature_keys:
            if key in features_dict:
                value = features_dict[key]
                # NaNや無限大値の処理
                if np.isnan(value) or np.isinf(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
            else:
                features.append(0.0)

        # パディング（フル特徴量と同じ次元にする）
        while len(features) < 10:  # フル特徴量は10次元
            features.append(0.0)

        return features

def train(self, X, y):
    """モデルを訓練する"""
    try:
        import streamlit as st

        # ストリームリットのセッション状態を使用して進捗を表示
        st.write(f"**訓練データ詳細**")
        st.write(f"- 総サンプル数: {len(X)}")

        # 各クラスの数を確認
        unique_labels, counts = np.unique(y, return_counts=True)
        st.write("- クラス別データ数:")
        for label, count in zip(unique_labels, counts):
            st.write(f"  - {label}: {count}個")
        
        # データの検証
        if len(X) == 0 or len(y) == 0:
            st.error("訓練データが空です")
            logger.error("訓練データが空です")
            return False


        # NaNや無限大値の処理
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        st.write(f"データクリーニング完了: {X_clean.shape}")


        # データの標準化
        X_scaled = self.scaler.fit_transform(X_clean)
        st.write(f"特徴量の標準化完了")


        # ランダムフォレスト分類器を作成
        self.model = RandomForestClassifier(
            n_estimators=100,  # 決定木の数
            max_depth=10,      # 木の最大深さ
            random_state=42,   # 再現性のための乱数シード
            n_jobs=-1          # 並列処理を使用
        )
        st.write(f"ランダムフォレストモデル作成: 100本の決定木を準備")


        # モデルの訓練
        st.write("**学習を開始しています...**")
        self.model.fit(X_scaled, y)
        st.write("**学習完了！**")


        # 訓練データでの精度を確認
        train_accuracy = self.model.score(X_scaled, y)
        st.success(f"訓練データでの精度: {train_accuracy:.1%}")


        # クラスのリストを保存
        self.classes = self.model.classes_
        st.write(f"学習したクラス: {list(self.classes)}")


        # 特徴量の重要度を取得
        importances = self.model.feature_importances_
        if self.feature_names is None:
            feature_names = [
                '平均音量', '音量変動', '文頭音量', '文中音量', '文末音量',
                '音量低下率', '最後20%音量', '最後20%低下率', 'スペクトル重心', '話速'
            ]
        else:
            feature_names = self.feature_names


        st.write("**特徴量の重要度:**")
        importance_data = []
        for name, importance in zip(feature_names[:len(importances)], importances):
            importance_data.append([name, f"{importance:.3f}"])

            
        # 表形式で表示
        importance_df = pd.DataFrame(importance_data, columns=['特徴量', '重要度'])
        st.dataframe(importance_df)


        # 訓練済みフラグを設定
        self.is_trained = True

        logger.info(f"モデル訓練完了: {len(X)}サンプル, {len(self.classes)}クラス")
        return True

    except Exception as e:
        st.logger.error(f"モデル訓練エラー: {e}")
        logger
        return False
    
    def predict(self, features_dict, realtime=False):
        """音声品質を予測する"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("モデルが訓練されていません")
                return None, 0
        
            # 特徴量配列を作成
            if realtime:
                features = self.prepare_features_realtime(features_dict)
            else:
                features = self.prepare_features(features_dict)
        
            # 特徴量を2次元配列に変換（sklearn要件）
            features_2d = np.array([features])
        
            # NaNや無限大値の処理
            features_2d = np.nan_to_num(features_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
            # 特徴量を標準化
            features_scaled = self.scaler.transform(features_2d)
        
            # 予測実行
            prediction = self.model.predict(features_scaled)[0]
        
            # 予測確率（信頼度）を取得
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        
            return prediction, confidence

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return None, 0

    def get_feature_importance(self):
        """特徴量の重要度を取得"""
        try:
            if not self.is_trained or self.model is None:
                return None

            importances = self.model.feature_importances_
            
            if self.feature_names is not None:
                feature_importance = dict(zip(self.feature_names, importances))
                return feature_importance
            else:
                return dict(zip([f'feature_{i}' for i in range(len(importances))], importances))

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return None

    def save_model(self, file_path):
        """モデルを保存する"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("保存するモデルがありません")
                return False
            # モデル情報を辞書にまとめる
            model_info = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'classes': self.classes,
                'feature_names': self.feature_names
            }

            # ファイルに保存
            joblib.dump(model_info, file_path)
            logger.info(f"モデルを保存しました: {file_path}")
            return True

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            return False

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
            self.feature_names = model_info.get('feature_names', None)
        
            logger.info(f"モデルを読み込みました: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False    

def generate_training_data():
    """機械学習用のシミュレーションデータを生成する関数"""  
    try:
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

        logger.info(f"訓練データを生成しました: {len(X)}サンプル")
        return np.array(X), np.array(y)
    
    except Exception as e:
        logger.error(f"訓練データ生成エラー: {e}")
        return np.array([]), np.array([])    

def create_dataset_from_files(file_paths):
    """音声ファイルからデータセットを作成する関数（将来の拡張用）"""
    """
    この関数は将来、実際の音声ファイルから特徴量を抽出して
    データセットを作成するために使用できます。
    現在はプレースホルダーとして空の実装になっています。
    """
    try:
        # 将来の実装:
        # 1. 各音声ファイルから特徴量を抽出
        # 2. ラベルを設定（ファイル名やメタデータから）
        # 3. 特徴量とラベルをまとめてデータセットを作成
        
        logger.info("音声ファイルからのデータセット作成は今後実装予定です")
        return np.array([]), np.array([])
        
    except Exception as e:
        logger.error(f"ファイルからのデータセット作成エラー: {e}")
        return np.array([]), np.array([])

# リアルタイム音声品質評価用のヘルパー関数
def quick_quality_assessment(features_dict):
    """軽量な音声品質評価（機械学習なし）"""
    try:
        drop_rate = features_dict.get('end_drop_rate', 0)
        mean_volume = features_dict.get('mean_volume', 0)
        
        # 簡単なルールベース評価
        if drop_rate < 0.15 and mean_volume > 0.05:
            return "良好", 0.9
        elif drop_rate < 0.3:
            return "普通", 0.7
        else:
            return "要改善", 0.5
            
    except Exception as e:
        logger.error(f"品質評価エラー: {e}")
        return "評価不可", 0.0