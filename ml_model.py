import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import pandas as pd
import streamlit as st

# ロガーの設定
logger = logging.getLogger(__name__)
# 必要なクラスと関数を実装
class VoiceQualityModel:
    """音声品質を評価する機械学習モデル

    このクラスは音声の特徴量から会話音声の品質を判断するAIモデルです。
    主に「良好」「文末が弱い」「小声すぎる」の3つのカテゴリに分類します。
    """

    def __init__(self):
        """モデルの初期化"""
        self.model = None # ランダムフォレストモデル
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = None

        # 特徴量の名前（日本語で表示用）
        self.feature_names = [
            '平均音量', '音量変動', '文頭音量', '文中音量', '文末音量',
            '音量低下率', '最後20%音量', '最後20%低下率', 'スペクトル重心', '話の速度'
        ]
        # モデルの性能記録用
        self.training_accuracy = 0
        self.test_accuracy = 0

    def prepare_features(self, features_dict):
        """特徴辞書から機械学習用の特徴量配列を作成"""
        # 使用する特徴量のキー：voice_analysis.py からの辞書
        feature_keys = [
            'mean_volume', 
            'std_volume', 
            'start_volume', 
            'middle_volume', 
            'end_volume',
            'end_drop_rate',
            'last_20_percent_volume',
            'last_20_percent_drop_rate',
            'spectral_centroid_mean',
            'speech_rate'
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
      
    def train(self, X, y):
        """モデルを訓練する
        引数:
            X (np.ndarray): 特徴量
            y (np.ndarray): ラベル
        Returns:
            bool: 訓練が成功したかどうか
        """
        try:
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
            
            # データを訓練用とテスト用に分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            st.write(f"訓練データとテストデータに分割: {len(X_train)}訓練, {len(X_test)}テスト")

            # NaNや無限大値の処理
            X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            st.write(f"データクリーニング完了: 訓練{X_train_clean.shape}, テスト{X_test_clean.shape}")

            # データの標準化(平均0, 標準偏差1)
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            X_test_scaled = self.scaler.transform(X_test_clean)
            st.write(f"特徴量の標準化が完了")

            # ランダムフォレスト分類器を作成
            self.model = RandomForestClassifier(
                n_estimators=200,  # 決定木の数
                max_depth=15,      # 木の最大深さ
                min_samples_split=5,  # 分割に必要な最小サンプル数
                min_samples_leaf=2,   # 葉に必要な最小サンプル数
                random_state=42,   # 再現性のための乱数シード
                n_jobs=-1,         # 並列処理を使用
                class_weight='balanced'  # クラスの不均衡を考慮
            )
            st.write(f"🤖 ランダムフォレストモデル作成: 200本の決定木を準備")

            # モデルの訓練
            st.write("**学習を開始しています...**")
            self.model.fit(X_train_scaled, y_train)
            st.write("**学習が完了しました！**")

            # 訓練データとテストデータでの精度を確認
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)

            st.success(f"訓練データでの精度: {train_accuracy:.1%}")
            st.success(f"テストデータでの精度: {test_accuracy:.1%}")

            # テストデータでの詳細な評価
            y_pred = self.model.predict(X_test_scaled)

            # 分類レポートを表示
            st.write("**詳細な評価結果:**")
            report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)

            # 各クラスの性能を表示
            for class_name in self.model.classea_:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1_score = report[class_name]['f1-score']
                    st.write(f"- **{class_name}**: 適合率={precision:.2f}, 再現率={recall:.2f}, F1={f1_score:.2f}")

            # 全体の性能を表示(マクロ平均)
            macro_avg = report['macro avg']
            weighted_avg = report['weighted avg']

            st.write("**全体の性能を表示(マクロ平均)**:")
            st.write(f"- **マクロ平均**: 精度={macro_avg['precision']:.2f}, 再現率={macro_avg['recall']:.2f}, F1スコア={macro_avg['f1-score']:.2f}")
            st.write(f"- **加重平均**: 精度={weighted_avg['precision']:.2f}, 再現率={weighted_avg['recall']:.2f}, F1スコア={weighted_avg['f1-score']:.2f}")

            # 性能の解釈
            if macro_avg['f1-score'] >= 0.8:
                st.success("モデルの性能は良好です！各クラスをバランスよく予測できています。")
            elif macro_avg['f1-score'] >= 0.7:
                st.warning("モデルの性能は普通です。実用なレベルに達しています。")
            elif macro_avg['f1-score'] >= 0.6:
                st.warning("モデルの性能は普通です。改善の余地があります。")
            else:
                st.error("モデルの性能は低いです。データの質や量を見直す必要があります。")               
                            
            # クラスのリストを保存
            self.classes = self.model.classes_
            st.write(f"学習したクラス: {list(self.classes)}")

           # 特徴量の重要度を取得
            importances = self.model.feature_importances_
            st.write(f"特徴量の重要度: {importances}")
            importance_data = []
            for name, importance in zip(self.feature_names[:len(importances)], importances):
                importance_data.append([name, f"{importance:.3f}"])

            # 表形式で表示
            importance_df = pd.DataFrame(importance_data, columns=['特徴量', '重要度'])
            importance_df = importance_df.sort_values(by='重要度', ascending=False)
            st.dataframe(importance_df)

            # 最も重要な特徴量を強調
            top_features = importance_df.iloc[:3]['特徴量']
            st.info(f"**最重要特徴量**: {', '.join(top_features)} - この特徴がAIの判断に最も影響しています")
            
            # 訓練済みフラグを設定
            self.is_trained = True

            logger.info(f"モデル訓練完了: {len(X)}サンプル, {len(self.classes)}クラス")
            return True
        
        except Exception as e:
            st.error(f"モデル訓練エラー: {e}")
            logger.error(f"モデル訓練エラー: {e}")
            return False
        
    def predict(self, features_dict, realtime=False):
        """音声品質を予測する
        引数:
            features_dict (dict): voice_analysis.py からの特徴量の辞書
        
        戻り値
            tuple: 予測結果と信頼度
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("モデルが訓練されていません")
                return None, 0
        
            # 特徴量配列を作成
            features = self.prepare_features_realtime(features_dict)

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
        """特徴量の重要度を取得
        戻り値:
            dict: 特徴量の重要度
        """
        try:
            if not self.is_trained or self.model is None:
                return None

            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return feature_importance
        
        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return None

    def get_model_performance(self):
        """モデルの性能を取得
        戻り値:
            dict: モデルの性能情報
        """
        if self.is_trained:
                return {
                    'training_accuracy': self.training_accuracy,
                    'test_accuracy': self.test_accuracy,
                    'feature_count': len(self.feature_names),
                    'class_count': len(self.classes) if self.classes is not None else 0
                }
        return None

    def save_model(self, file_path):
        """モデルを保存する
        引数:
            file_path (str): 保存先のファイルパス
        戻り値:
            bool: 保存成功ならTrue、失敗ならFalse   
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("保存するモデルがありません")
                return False
            
            model_info = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'classes': self.classes,
                'feature_names': self.feature_names,
                'training_accuracy': self.training_accuracy,
                'test_accuracy': self.test_accuracy
                }

            joblib.dump(model_info, file_path)
            logger.info(f"モデルを保存しました: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            return False
        
    def load_model(self,file_path):
        """保存されたモデルを読み込む
        引数:
        file_path (str): 読み込み元のファイルパス
        """
        try:
            # モデルを読み込む
            model_info = joblib.load(file_path)

            self.model = model_info['model']
            self.scaler = model_info['scaler']
            self.is_trained = model_info['is_trained']
            self.classes = model_info['classes']
            self.feature_names = model_info.get('feature_names', self.feature_names)
            self.training_accuracy = model_info.get('training_accuracy', 0)

            logger.info(f"モデルを読み込みました: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

# クラス外の独立した関数として定義
def generate_training_data():
    """機械学習用のシミュレーションデータを生成する関数
    引数:
        なし
    戻り値:
        tuple: 特徴量データとラベル
    """
    try:
        x = []  # 特徴量データ
        y = []  # ラベルデータ

        # シミュレーションデータを生成
        for i in range(80):
            features = [
                np.random.uniform(0.08, 0.25),   # mean_volume
                np.random.uniform(0.015, 0.04),   # std_volume
                np.random.uniform(0.08, 0.25),     # start_volume
                np.random.uniform(0.08, 0.25),     # middle_volume
                np.random.uniform(0.07, 0.22),   # end_volume（そこまで低下しない）
                np.random.uniform(0.03, 0.12),   # end_drop_rate（小さめ）
                np.random.uniform(0.07, 0.22),   # last_20_percent_volume
                np.random.uniform(0.03, 0.12),   # last_20_percent_drop_rate
                np.random.uniform(1200, 2200),   # spectral_centroid_mean
                np.random.uniform(2.5, 4.5),     # speech_rate
            ]

            x.append(features)
            y.append("良好")  

        # 「文末が弱い」音声のデータ
        for i in range(80):
            features = [
                np.random.uniform(0.08, 0.25),   # mean_volume
                np.random.uniform(0.015, 0.04),   # std_volume
                np.random.uniform(0.08, 0.25),     # start_volume
                np.random.uniform(0.08, 0.25),     # middle_volume
                np.random.uniform(0.02, 0.08),      # end_volume（明らかに低い）
                np.random.uniform(0.25, 0.6),      # end_drop_rate（大きい）
                np.random.uniform(0.02, 0.08),      # last_20_percent_volume
                np.random.uniform(0.25, 0.6),      # last_20_percent_drop_rate
                np.random.uniform(1000, 2000),     # spectral_centroid_mean
                np.random.uniform(2, 4),           # speech_rate
            ]
            x.append(features)
            y.append("文末が弱い")

        # 「小声すぎる」音声のデータ
        for _ in range(50): 
            features = [
                np.random.uniform(0.005, 0.04),  # mean_volume（全体的に小さい）
                np.random.uniform(0.005, 0.015), # std_volume
                np.random.uniform(0.005, 0.04),  # start_volume
                np.random.uniform(0.005, 0.04),  # middle_volume
                np.random.uniform(0.003, 0.025), # end_volume
                np.random.uniform(0.1, 0.35),    # end_drop_rate
                np.random.uniform(0.003, 0.025), # last_20_percent_volume
                np.random.uniform(0.1, 0.35),    # last_20_percent_drop_rate
                np.random.uniform(800, 1400),    # spectral_centroid_mean（低め）
                np.random.uniform(1.5, 3),       # speech_rate（遅め）        
            ]
            x.append(features)
            y.append("小声すぎる")

        logger.info(f"訓練データを生成しました: {len(x)}サンプル")
        return np.array(x), np.array(y)

    except Exception as e:
        logger.error(f"訓練データ生成エラー: {e}")
        return np.array([]), np.array([])

# データセット保存・読み込み機能
def save_training_data(X, y, file_path):
    """訓練データを保存
    引数:
        X (np.ndarray): 特徴量
        y (np.ndarray): ラベル
        file_path (str): 保存先のファイルパス
    戻り値:
        bool: 保存成功ならTrue、失敗ならFalse
    """
    try:
        np.savez(file_path, X=X, y=y)
        logger.info(f"訓練データを保存しました: {file_path}")
        return True
    except Exception as e:
        logger.error(f"訓練データ保存エラー: {e}")
        return False

def load_training_data(file_path):
    """訓練データを読み込む
    引数:
        file_path (str): 読み込み元のファイルパス
    戻り値:
        tuple: 特徴量データとラベル
        """
    try:
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        logger.info(f"訓練データを読み込みました: {file_path}")
        return X, y
    except Exception as e:
        logger.error(f"訓練データ読み込みエラー: {e}")
        return np.array([]), np.array([])
    
# リアルタイム音声品質評価用のヘルパー関数
def quick_quality_assessment(features_dict):
    """軽量な音声品質評価（機械学習なし）
    機械学習モデルが利用できない場合の簡易評価。
    ルールベースで会話の品質を評価します。

    引数:
        features_dict (dict): 音声特徴量の辞書
    戻り値:
        tuple: 評価結果と信頼度
        """
    try:
        drop_rate = features_dict.get('end_drop_rate', 0)
        mean_volume = features_dict.get('mean_volume', 0)
        
        # より詳細なルールベース評価
        if drop_rate < 0.1 and mean_volume > 0.05:
            return "良好", 0.95
        elif drop_rate < 0.15 and mean_volume > 0.03:
            return "良好", 0.85
        elif drop_rate < 0.25 and mean_volume > 0.02:
            return "普通", 0.7
        elif drop_rate < 0.4:
            return "文末が弱い", 0.6
        elif mean_volume < 0.02:
            return "小声すぎる", 0.5
        else:
            return "会話をもう少し意識してみましょう", 0.4
            
    except Exception as e:
        logger.error(f"品質評価エラー: {e}")
        return "評価不可", 0.0
    
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