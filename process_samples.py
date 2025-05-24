import os
import numpy as np
import librosa
import pandas as pd
import streamlit as st
from voice_analysis import VoiceFeatureExtractor
from voice_analysis import plot_audio_analysis

# process_samples.py
import matplotlib.pyplot as plt

# 設定
DATA_DIR = 'data'
CATEGORIES = ['good', 'weak_ending']  # カテゴリフォルダ名
OUTPUT_FILE = 'features_data.csv'  # 特徴量を保存するCSVファイル

def extract_features_from_file(audio_path):
    """音声ファイルから特徴量を抽出する"""
    try:
        # 音声の読み込み
        y, sr = librosa.load(audio_path, sr=None)
        
        # 特徴抽出器を使用して特徴量を取得
        extractor = VoiceFeatureExtractor()
        features = extractor.extract_features(y, sr)
        
        return features, y, sr
    except Exception as e:
        print(f"エラー: {audio_path} の処理中に問題が発生しました: {e}")
        return None, None, None

def analyze_audio_sample(audio_path, output_dir='analysis'):
    """音声サンプルを分析しグラフを生成する"""
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名（拡張子なし）の取得
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 特徴量抽出
    features, y, sr = extract_features_from_file(audio_path)
    if features is None:
        return None
    
    # 波形と音量変化のグラフを生成
    fig = plot_audio_analysis(features, y, sr)
    
    # グラフを保存
    plt.savefig(os.path.join(output_dir, f'{filename}_analysis.png'))
    plt.close(fig)
    
    return features

def process_all_samples():
    """すべての音声サンプルを処理してCSVに保存する"""
    all_features = []
    
    # 各カテゴリのサンプルを処理
    for category in CATEGORIES:
        category_dir = os.path.join(DATA_DIR, category)
        
        # カテゴリディレクトリが存在するか確認
        if not os.path.exists(category_dir):
            print(f"警告: ディレクトリ {category_dir} が見つかりません。スキップします。")
            continue
        
        # カテゴリ内の全音声ファイルを処理
        for filename in os.listdir(category_dir):
            if filename.endswith(('.wav', '.mp3')):
                file_path = os.path.join(category_dir, filename)
                
                print(f"処理中: {file_path}")
                
                # 音声サンプルの分析
                features = analyze_audio_sample(file_path)
                if features:
                    # カテゴリラベルの追加
                    features['category'] = category
                    
                    # 結果を追加
                    all_features.append(features)
    
    # 特徴量をDataFrameに変換
    if all_features:
        df = pd.DataFrame(all_features)
        
        # CSVファイルに保存
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"{len(df)}個のサンプルの特徴量を {OUTPUT_FILE} に保存しました。")
        
        # 特徴量の統計情報を表示
        print("\n特徴量の統計情報:")
        print(df.describe())
        
        # カテゴリごとの平均値を表示
        print("\nカテゴリごとの平均値:")
        print(df.groupby('category').mean(numeric_only=True))
    else:
        print("有効なサンプルが見つかりませんでした。")

if __name__ == "__main__":
    process_all_samples()