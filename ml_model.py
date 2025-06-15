import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import pandas as pd
import sys
import traceback


# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceQualityModel:
    """音声品質を評価する機械学習モデル"""
    
    def __init__(self):
        """モデルの初期化"""
        self.model = None # ランダムフォレストモデル
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = None

        # 特徴量の名前（日本語で表示用）
        self.feature_names = [
            'Mean Volume(平均音量)', 'Volume Variation(音量変動)', 'Start Volume(文頭音量)', 'Middle Volume(文中音量)', 'End Volume(文末音量)',
            'Volume Drop Rate(音量低下率)', 'Last 20% Volume(最後20%音量)', 'Last 20% Drop Rate(最後20%低下率)', 'Spectral Centroid(スペクトル重心)', 'Speech Rate(話の速度)'
        ]
        
        # モデルの性能記録用
        self.training_accuracy = 0
        self.test_accuracy = 0

    def prepare_features(self, features_dict):
        """特徴辞書から機械学習用の特徴量配列を作成"""
        # 使用する特徴量のキーからの辞書
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
        """モデルを訓練する"""
        try:
            #データの検証（最初に実行）
            if len(X) == 0 or len(y) ==0:
                st.error("訓練データが空です")
                logger.error(f"訓練データが空です")
                return False
            
            st.write("**AI訓練を開始します**")
            st.write(f"総サンプル数: {len(X)}")           

            # 各クラスの数を確認
            unique_labels, counts = np.unique(y, return_counts=True)
            st.write("クラス別データ数:")
            for label, count in zip(unique_labels, counts):
                st.write(f"  - {label}: {count}個")
            
            # データを訓練用とテスト用に分割
            st.write("データを訓練用とテスト用に分割中...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            st.write(f"データ分割完了: 訓練{len(X_train)}個, テスト{len(X_test)}個")
            
            # データクリーニング (NaNや無限大値の処理)
            st.write("データクリーニング中...")
            X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            st.write(f"データクリーニング完了")

            # データの標準化(平均0, 標準偏差1)
            st.write("特徴量の標準化中...")
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            X_test_scaled = self.scaler.transform(X_test_clean)
            st.write(f"特徴量の標準化が完了")

            # ランダムフォレスト分類器を作成
            st.write("AIモデル作成中...")
            self.model = RandomForestClassifier(
                n_estimators=100,  # 決定木の数
                max_depth=10,      # 木の最大深さ
                min_samples_split=5,  # 分割に必要な最小サンプル数
                min_samples_leaf=2,   # 葉に必要な最小サンプル数
                random_state=42,   # 再現性のための乱数シード
                n_jobs=-1,         # 並列処理を使用
                class_weight='balanced'  # クラスの不均衡を考慮
            )
            st.write(f"🤖 ランダムフォレストモデル作成: 100本の決定木を準備")

            # モデルの訓練
            st.write("**学習を開始しています...**")
            
            # プログレスバーを表示
            progress_bar = st.progress(0)
            progress_bar.progress(50)            
            
            self.model.fit(X_train_scaled, y_train)
            st.write("**AI学習が完了しました！**")

            # 訓練データとテストデータでの精度を確認
            st.write("精度評価中...")
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)

            st.success(f"訓練データでの精度: {train_accuracy:.1%}")
            st.success(f"テストデータでの精度: {test_accuracy:.1%}")

            # クラス情報を保存
            self.classes = self.model.classes_
            self.is_trained = True
            self.training_accuracy = train_accuracy
            self.test_accuracy = test_accuracy
            
            st.write(f"学習したクラス: {list(self.classes)}")

            # テストデータでの詳細な評価
            st.write("詳細な評価結果を計算中...")
            y_pred = self.model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, target_names=unique_labels, output_dict=True)

            # 各クラスの性能を表示
            st.write("**各クラスの性能:**")
            for class_name in unique_labels:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1_score = report[class_name]['f1-score']
                    st.write(f"- **{class_name}**: 適合率={precision:.2f}, 再現率={recall:.2f}, F1={f1_score:.2f}")

            # 全体の性能を表示(マクロ平均)
            macro_avg = report['macro avg']
            st.write(f"**全体F1スコア**: {macro_avg['f1-score']:.2f}")          
            
            # 性能の解釈
            if macro_avg['f1-score'] >= 0.8:
                st.success("モデルの性能は良好です！各クラスをバランスよく予測できています。")
            elif macro_avg['f1-score'] >= 0.7:
                st.info("モデルの性能は普通です。実用なレベルに達しています。")
            elif macro_avg['f1-score'] >= 0.6:
                st.warning("モデルの性能は普通です。改善の余地があります。")
            else:
                st.error("モデルの性能は低いです。データの質や量を見直す必要があります。")               
                            
            # 特徴量の重要度を取得
            st.write("特徴量の重要度を分析中...")
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                '特徴量': self.feature_names[:len(importances)],
                '重要度': [f"{imp:.3f}" for imp in importances]
            }).sort_values('重要度', ascending=False)

            st.write("**特徴量の重要度:**")
            st.dataframe(importance_df)

            # 最重要特徴量を強調
            top_features = importance_df.iloc[:3]['特徴量'].tolist()
            st.info(f"**最重要特徴量**: {', '.join(top_features)} - この特徴がAIの判断に最も影響しています")

            # プログレスバーをクリア
            progress_bar.empty()
            
            st.success("**AI訓練が完全に完了しました！** これで「練習を始める」ページで高精度な音声分析が利用できます。")
            
            logger.info(f"モデル訓練完了: {len(X)}サンプル, {len(self.classes)}クラス")            
            return True         
                  
        except Exception as e:
            st.error(f"AI訓練エラー: {e}")
            logger.error(f"訓練エラー: {e}")
            return False            
                       
    def predict(self, features_dict):
        """音声品質を予測する"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("モデルが訓練されていません")
                return None, 0
        
            # 特徴量配列を作成
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
            feature_importance = dict(zip(self.feature_names, importances))
            return feature_importance
        
        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return None

    def get_model_performance(self):
        """モデルの性能を取得"""
        if self.is_trained:
                return {
                    'training_accuracy': self.training_accuracy,
                    'test_accuracy': self.test_accuracy,
                    'feature_count': len(self.feature_names),
                    'class_count': len(self.classes) if self.classes is not None else 0
                }
        return None

# クラス外の独立した関数として定義
def generate_training_data():
    """機械学習用のシミュレーションデータを生成する関数"""
    try:
        x = []  # 特徴量データ
        y = []  # ラベルデータ

        # シミュレーションデータを生成
        # 「良好」な音声データ（80個）
        for i in range(70):
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

            # ノイズを追加（現実的なバラツキを再現）
            noise = np.random.normal(0, 0.01, len(features))
            features = np.array(features) + noise


            x.append(features)
            y.append("良好")  

        # 「文末が弱い」音声のデータ
        for i in range(70):
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
            # ノイズを追加（現実的なバラツキを再現）
            noise = np.random.normal(0, 0.01, len(features))
            features = np.array(features) + noise

            
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

            # ノイズを追加
            noise = np.random.normal(0, 0.008, len(features))  # 少し弱めのノイズ
            features = np.array(features) + noise


            x.append(features)
            y.append("小声すぎる")

        logger.info(f"訓練データを生成しました: {len(x)}サンプル")
        return np.array(x), np.array(y)

    except Exception as e:
        logger.error(f"訓練データ生成エラー: {e}")
        return np.array([]), np.array([])
   
def quick_quality_assessment(features_dict):
    """軽量な音声品質評価（機械学習なし）"""
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
    
# def create_dataset_from_files(file_paths):
    # """音声ファイルからデータセットを作成する関数（将来の拡張用）"""
    # """
    # この関数は将来、実際の音声ファイルから特徴量を抽出して
    # データセットを作成するために使用できます。
    # 現在はプレースホルダーとして空の実装になっています。
    # """
    # try:
        # 将来の実装:
        # 1. 各音声ファイルから特徴量を抽出
        # 2. ラベルを設定（ファイル名やメタデータから）
        # 3. 特徴量とラベルをまとめてデータセットを作成
        
    #    logger.info("音声ファイルからのデータセット作成は今後実装予定です")
    #    return np.array([]), np.array([])
        
    #except Exception as e:
    #    logger.error(f"ファイルからのデータセット作成エラー: {e}")
    #    return np.array([]), np.array([])

