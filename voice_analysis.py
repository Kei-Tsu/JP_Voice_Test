# ml_model.py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

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
        if third > 0:
            features['start_volume'] = np.mean(rms[:third])  # 最初の1/3
            features['middle_volume'] = np.mean(rms[third:2*third])  # 中間の1/3
            features['end_volume'] = np.mean(rms[2*third:])  # 最後の1/3
        else:
            # 音声が短すぎる場合は全体の平均を使用
            features['start_volume'] = features ['mean_volume']
            features['middle_volume'] = features ['mean_volume']
            features['end_volume'] = features ['mean_volume']

        # 文末音量低下率の計算
        features['end_drop_rate'] = (features['middle_volume'] - features['end_volume']) / features['middle_volume'] if features['middle_volume'] > 0 else 0
        
        # より詳細な文末分析（最後の20%部分）
        end_portion = max(1, int(len(rms) * 0.2))  # 最後の20%
        features['last_20_percent_volume'] = np.mean(rms[-end_portion:])
        features['last_20_percent_drop_rate'] = (features['mean_volume'] - features['last_20_percent_volume']) / features['mean_volume'] if features['mean_volume'] > 0 else 0
        
        # MFCC特徴量（音声の音色特性）
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(len(mfccs)):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        except Exception as e:
            # MFCC抽出に失敗した場合はデフォルト値を設定
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0

        # スペクトル特性
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        except Exception as e:
            # スペクトル特性抽出に失敗した場合はデフォルト値を設定
            features['spectral_centroid_mean'] = 0.0    
        
        # 音声のペース（オンセット検出で音節を近似）
        try:
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            features['onset_count'] = len(onsets)
            features['speech_rate'] = len(onsets) / (len(audio_data) / sr) if len(audio_data) > 0 else 0
        except Exception as e:
            # オンセット検出に失敗した場合はデフォルト値を設定
            features['onset_count'] = 0
            features['speech_rate'] = 0

        return features

    def extract_realtime_features(self, audio_segment):
        """リアルタイム用の軽量特徴量抽出"""
        try:
            # pydub AudioSegmentからnumpy配列に変換
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
                audio_data = audio_data.mean(axis=1)  # ステレオをモノラルに変換
            
            # 正規化
            if len(audio_data) > 0:
                audio_data = audio_data / (2**15)  # 16bit音声として正規化
            
            # サンプルレート
            sr = audio_segment.frame_rate
            
            # 基本的な特徴量のみ抽出（処理速度優先）
            features = {}
            
            # RMS音量
            if len(audio_data) > 0:
                rms = librosa.feature.rms(y=audio_data)[0]
                features['mean_volume'] = np.mean(rms)
                features['std_volume'] = np.std(rms)
                
                # 簡易的な文末分析
                third = len(rms) // 3
                if third > 0:
                    features['end_volume'] = np.mean(rms[2*third:])
                    features['middle_volume'] = np.mean(rms[third:2*third])
                    features['end_drop_rate'] = (features['middle_volume'] - features['end_volume']) / features['middle_volume'] if features['middle_volume'] > 0 else 0
                else:
                    features['end_volume'] = features['mean_volume']
                    features['middle_volume'] = features['mean_volume']
                    features['end_drop_rate'] = 0
            else:
                features['mean_volume'] = 0
                features['std_volume'] = 0
                features['end_volume'] = 0
                features['middle_volume'] = 0
                features['end_drop_rate'] = 0
            
            return features
        except Exception as e:
            # エラーが発生した場合はデフォルト値を返す
            return {
                'mean_volume': 0,
                'std_volume': 0,
                'end_volume': 0,
                'middle_volume': 0,
                'end_drop_rate': 0
            }

def plot_audio_analysis(features, audio_data, sr):
    """音声分析の視覚化を行う関数"""
    try:
        # 2つのサブプロットを作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
        # 1つ目のプロット: 波形表示
        librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
        ax1.set_title('Audio Waveform')
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Amplitude')
    
        # 2つ目のプロット: 音量変化
        if 'rms' in features and 'times' in features:
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
            if 'start_volume' in features:
                ax2.axhline(y=features['start_volume'], color='green', linestyle='--', label='Start Volume Average')
            if 'middle_volume' in features:
                ax2.axhline(y=features['middle_volume'], color='orange', linestyle='--', label='Middle Volume Average')
            if 'end_volume' in features:
                ax2.axhline(y=features['end_volume'], color='red', linestyle='--', label='End Volume Average')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '音声データがありません', fontsize=12, ha='center')
            ax2.set_title('Volume Change Over Time')

        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'音声分析エラー: {e}', fontsize=12, ha='center')
        ax.set_title('音声分析エラー')
        ax.set_xlabel('音声データがありません')
        plt.tight_layout()
        return fig


def evaluate_clarity(features):
    """音量特徴からクリアな発話かどうかを評価する関数"""
    try:
        drop_rate = features.get("end_drop_rate", 0)
        last_20_drop_rate = features.get("last_20_percent_drop_rate", 0)  # キーがない場合は0

        # 両方のドロップ率を考慮した評価
        avg_drop_rate = (drop_rate + last_20_drop_rate) / 2

        if avg_drop_rate < 0.1:
            clarity_level = "良好"
            advice = "語尾までしっかり発話できています！バランスがよい発話です。"
            score = min(100, int((1 - avg_drop_rate) * 100))
        elif avg_drop_rate < 0.25:
            clarity_level = "普通"
            advice = "語尾がやや弱まっています。少しだけ意識をして話すと良いでしょう。"
            score = int(75 - (avg_drop_rate - 0.1) * 100)
        elif avg_drop_rate < 0.4:
            clarity_level = "やや弱い"
            advice = "文末の音量がやや弱めです。文末を1音上げるイメージで話してみましょう。"
            score = int(60 - (avg_drop_rate - 0.25) * 100)
        else:
            clarity_level = "少し頑張りましょう"
            advice = "語尾の音量が低下しています。結論が正しく伝わるよう意識して話してみましょう。"
            score = max(20, int(40 - (avg_drop_rate - 0.4) * 50))
    
        return {
            "clarity_level": clarity_level,
            "advice": advice,
            "score": score,
            "avg_drop_rate": avg_drop_rate
        }
    except Exception as e:
        # エラーが発生した場合はデフォルト値を返す
        return {
            "clarity_level": "評価ができません",
            "advice": "音声の分析中にエラーが発生しました。",
            "score": 0,
            "avg_drop_rate": 0
        }   

# ドロップ率に応じたフィードバックを生成
def get_feedback(drop_rate):
    """ドロップ率に基づいてフィードバックを生成する関数"""
    try:
        if drop_rate < 0.1:
            return {
                "level": "good",
                "message": "良い感じです！語尾までしっかり発音できています。",
                "emoji": "🌟"
        }
        elif drop_rate < 0.25:
            return {
                "level": "medium",
                "message": "語尾がやや弱まっています。少しだけ意識しましょう。",
                "emoji": "⚠️"
             }
        else:
            return {
                "level": "bad",
                "message": "語尾の音量が大きく低下しています。文末を意識して！",
                "emoji": "❗"
            }
    except Exception as e:
        return {
            "level": "error",
            "message": "フィードバック生成中にエラーが発生しました。",
            "emoji": "❓"
        }

# リアルタイム分析のための補助関数
def analyze_volume_realtime(audio_segment):
    """リアルタイム音量分析"""
    try:
        extractor = VoiceFeatureExtractor()
        features = extractor.extract_realtime_features(audio_segment)
        evaluation = evaluate_clarity(features)
        return features, evaluation
    except Exception as e:
        # エラーが発生した場合はデフォルト値を返す
        default_features = {
            'mean_volume': 0,
            'std_volume': 0,
            'end_volume': 0,
            'middle_volume': 0,
            'end_drop_rate': 0
        }
        default_evaluation = {
            'clarity_level': '評価不可',
            'advice': 'リアルタイム分析でエラーが発生しました。',
            'score': 0,
            'avg_drop_rate': 0
        }
        return default_features, default_evaluation
