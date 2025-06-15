# ml_model.py
# 日本語フォント対応
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

class VoiceFeatureExtractor:
    """音声から特徴量を抽出するクラス（マイク補正機能付き）"""

    def __init__(self):
        # 親密な会話の音量範囲（dB SPL基準）
        self.intimate_conversation_range = {
            'min_db': -45,  # 小声の下限
            'max_db': -25,  # 小声の上限
            'target_db': -35  # 目標音量
        }

    def normalize_for_intimate_conversation(self, audio_data, recording_source="file"):
        """親密な会話レベルに音量を正規化"""
        try:
            # RMS音量を計算
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms == 0:
                return audio_data
            
            # 現在の音量レベル（dB）
            current_db = 20 * np.log10(rms + 1e-10)
            
            # 録音ソースに応じた補正
            if recording_source == "microphone":
                # マイク録音の場合：より大きな補正を適用
                target_db = self.intimate_conversation_range['target_db']
                # マイクによる音量増幅を考慮した補正係数
                mic_compensation = -10  # マイクによる10dB増幅を想定
                target_db += mic_compensation
            elif recording_source == "realtime":
                # リアルタイム録音の場合：中程度の補正
                target_db = self.intimate_conversation_range['target_db'] - 5
            else:
                # ファイルアップロードの場合：軽微な補正
                target_db = self.intimate_conversation_range['target_db']
            
            # 正規化比率を計算
            db_difference = target_db - current_db
            scale_factor = 10**(db_difference / 20)
            
            # 音量を調整
            normalized_audio = audio_data * scale_factor
            
            # クリッピング防止
            max_amplitude = np.max(np.abs(normalized_audio))
            if max_amplitude > 0.95:
                normalized_audio = normalized_audio * (0.95 / max_amplitude)
            
            return normalized_audio
            
        except Exception as e:
            st.warning(f"音量正規化中にエラー: {e}")
            return audio_data

    def extract_features(self, audio_data, sr, recording_source="file"):
        """音声特徴量を抽出する関数（録音ソース考慮）"""
        # まず音量を親密な会話レベルに正規化
        normalized_audio = self.normalize_for_intimate_conversation(audio_data, recording_source)
        
        features = {}
        features['recording_source'] = recording_source  # 録音ソース情報を保存
        
        # 基本的な音量特徴量（RMS）
        rms = librosa.feature.rms(y=normalized_audio)[0]
        times = librosa.times_like(rms, sr=sr)
        features['rms'] = rms
        features['times'] = times
        features['mean_volume'] = np.mean(rms)
        features['std_volume'] = np.std(rms)
        features['max_volume'] = np.max(rms)
        features['min_volume'] = np.min(rms)
        
        # 元の音声との比較情報
        original_rms = librosa.feature.rms(y=audio_data)[0]
        features['original_mean_volume'] = np.mean(original_rms)
        features['volume_adjustment_ratio'] = features['mean_volume'] / (features['original_mean_volume'] + 1e-10)
        
        # 会話音声を三分割した分析
        third = len(rms) // 3
        if third > 0:
            features['start_volume'] = np.mean(rms[:third])
            features['middle_volume'] = np.mean(rms[third:2*third])
            features['end_volume'] = np.mean(rms[2*third:])
        else:
            features['start_volume'] = features['mean_volume']
            features['middle_volume'] = features['mean_volume']
            features['end_volume'] = features['mean_volume']

        # 文末音量低下率の計算
        features['end_drop_rate'] = (features['middle_volume'] - features['end_volume']) / features['middle_volume'] if features['middle_volume'] > 0 else 0
        
        # より詳細な文末分析（最後の20%部分）
        end_portion = max(1, int(len(rms) * 0.2))
        features['last_20_percent_volume'] = np.mean(rms[-end_portion:])
        features['last_20_percent_drop_rate'] = (features['mean_volume'] - features['last_20_percent_volume']) / features['mean_volume'] if features['mean_volume'] > 0 else 0
        
        # 親密さレベルの評価
        features['intimacy_level'] = self.assess_intimacy_level(features, recording_source)
        
        # MFCC特徴量
        try:
            mfccs = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=13)
            for i in range(len(mfccs)):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        except Exception as e:
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0

        # スペクトル特性
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=normalized_audio, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        except Exception as e:
            features['spectral_centroid_mean'] = 0.0    
        
        # 音声のペース
        try:
            onset_env = librosa.onset.onset_strength(y=normalized_audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            features['onset_count'] = len(onsets)
            features['speech_rate'] = len(onsets) / (len(normalized_audio) / sr) if len(normalized_audio) > 0 else 0
        except Exception as e:
            features['onset_count'] = 0
            features['speech_rate'] = 0

        return features

    def assess_intimacy_level(self, features, recording_source):
        """親密さレベルを評価"""
        mean_vol = features['mean_volume']
        
        # 録音ソースに応じた閾値調整
        if recording_source == "microphone":
            # マイク録音の場合：より厳しい基準
            if mean_vol < 0.03:
                return "very_intimate"
            elif mean_vol < 0.08:
                return "intimate"
            elif mean_vol < 0.15:
                return "casual"
            else:
                return "formal"
        elif recording_source == "realtime":
            # リアルタイムの場合：中程度の基準
            if mean_vol < 0.05:
                return "very_intimate"
            elif mean_vol < 0.12:
                return "intimate"
            elif mean_vol < 0.20:
                return "casual"
            else:
                return "formal"
        else:
            # ファイルアップロードの場合：標準基準
            if mean_vol < 0.04:
                return "very_intimate"
            elif mean_vol < 0.10:
                return "intimate"
            elif mean_vol < 0.18:
                return "casual"
            else:
                return "formal"

    def extract_realtime_features(self, audio_segment):
        """リアルタイム用の軽量特徴量抽出（マイク補正付き）"""
        try:
            # pydub AudioSegmentからnumpy配列に変換
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
                audio_data = audio_data.mean(axis=1)
            
            # 正規化
            if len(audio_data) > 0:
                audio_data = audio_data / (2**15)
            
            sr = audio_segment.frame_rate
            
            # リアルタイム録音として正規化
            normalized_audio = self.normalize_for_intimate_conversation(audio_data, "realtime")
            
            features = {}
            
            if len(normalized_audio) > 0:
                rms = librosa.feature.rms(y=normalized_audio)[0]
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
                
                # 親密さレベル
                features['intimacy_level'] = self.assess_intimacy_level(features, "realtime")
            else:
                features = {
                    'mean_volume': 0, 'std_volume': 0, 'end_volume': 0,
                    'middle_volume': 0, 'end_drop_rate': 0, 'intimacy_level': 'unknown'
                }
            
            return features
        except Exception as e:
            return {
                'mean_volume': 0, 'std_volume': 0, 'end_volume': 0,
                'middle_volume': 0, 'end_drop_rate': 0, 'intimacy_level': 'unknown'
            }

#クラス外の関数
def plot_audio_analysis(features, audio_data, sr):
    """音声分析の視覚化を行う関数（補正情報付き）"""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
        # 1つ目のプロット: 波形表示
        librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
        ax1.set_title('Original Audio Waveform')
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Amplitude')
    
        # 2つ目のプロット: 音量変化
        if 'rms' in features and 'times' in features:
            rms = features['rms']
            times = features['times']
            ax2.plot(times, rms, color='blue', label='Normalized Volume (RMS)')
            ax2.set_title('Volume Change Over Time (Normalized for Intimate Conversation)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Volume (RMS)')
    
            # 文末部分を強調表示
            if len(times) > 0:
                end_portion = max(1, int(len(times) * 0.2))
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
        
        # 3つ目のプロット: 補正情報
        ax3.axis('off')
        recording_source = features.get('recording_source', 'unknown')
        intimacy_level = features.get('intimacy_level', 'unknown')
        adjustment_ratio = features.get('volume_adjustment_ratio', 1.0)
        
        info_text = f"""
音量補正情報:
• 録音ソース: {recording_source}
• 親密さレベル: {intimacy_level}
• 音量調整比率: {adjustment_ratio:.2f}
• 文末音量低下率: {features.get('end_drop_rate', 0):.3f}

親密さレベルの説明:
• very_intimate: とても親密（家族間のソフトな声）
• intimate: 親密（恋人同士の普通の声）
• casual: カジュアル（友人との会話）
• formal: フォーマル（公式な場での話し方）
        """

        ax3.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'音声分析エラー: {str(e)}', fontsize=12, ha='center')
        ax.set_title('音声分析結果')
        plt.tight_layout()
        return fig

def evaluate_clarity(features):
    """音量特徴からクリアな発話かどうかを評価する関数（録音ソース考慮）"""
    try:
        drop_rate = features.get("end_drop_rate", 0)
        last_20_drop_rate = features.get("last_20_percent_drop_rate", 0)
        recording_source = features.get("recording_source", "file")
        intimacy_level = features.get("intimacy_level", "casual")
        
        # 両方のドロップ率を考慮
        avg_drop_rate = (drop_rate + last_20_drop_rate) / 2
        
        # 録音ソースと親密さレベルに応じた評価基準の調整
        if recording_source == "microphone" and intimacy_level in ["intimate", "very_intimate"]:
            # マイク録音で親密な会話の場合：より緩い基準
            if avg_drop_rate < 0.15:
                clarity_level = "良好"
                advice = "親密な会話として語尾までしっかり発話できています！"
                score = min(100, int((1 - avg_drop_rate * 0.8) * 100))
            elif avg_drop_rate < 0.3:
                clarity_level = "普通"
                advice = "親密な会話レベルとしては普通です。実際の会話ではもう少しだけ語尾を意識すると良いでしょう。"
                score = int(75 - (avg_drop_rate - 0.15) * 80)
            elif avg_drop_rate < 0.45:
                clarity_level = "やや弱い"
                advice = "語尾がやや弱めです。親密な会話でも、文末を1音上げるイメージで話してみましょう。"
                score = int(60 - (avg_drop_rate - 0.3) * 80)
            else:
                clarity_level = "要改善"
                advice = "語尾の音量が大きく低下しています。親密な関係でも少しだけ声量に意識してみましょう。"
                score = max(20, int(40 - (avg_drop_rate - 0.45) * 40))
        else:
            # 標準的な評価基準
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
                clarity_level = "要改善"
                advice = "語尾の音量が低下しています。話し方も少し意識して話してみましょう。"
                score = max(20, int(40 - (avg_drop_rate - 0.4) * 50))
    
        return {
            "clarity_level": clarity_level,
            "advice": advice,
            "score": score,
            "avg_drop_rate": avg_drop_rate,
            "recording_context": f"{recording_source}録音, {intimacy_level}レベル"
        }
    except Exception as e:
        return {
            "clarity_level": "評価ができません",
            "advice": "音声の分析中にエラーが発生しました。",
            "score": 0,
            "avg_drop_rate": 0,
            "recording_context": "不明"
        }   

def get_feedback(drop_rate, intimacy_level="casual"):
    """ドロップ率と親密さレベルに基づいてフィードバックを生成"""
    try:
        # 親密さレベルに応じた閾値調整
        if intimacy_level in ["intimate", "very_intimate"]:
            # 親密な会話用の緩い基準
            if drop_rate < 0.15:
                return {
                    "level": "good",
                    "message": "親密な会話として良い感じです！語尾までしっかり聞こえています。",
                }
            elif drop_rate < 0.3:
                return {
                    "level": "medium",
                    "message": "親密な会話としては普通です。もう少し語尾を意識してみましょう。",
                }
            else:
                return {
                    "level": "bad",
                    "message": "親密な会話でも語尾が弱くなっています。最後まで声を届けましょう。",
                }
        else:
            # 標準的な基準
            if drop_rate < 0.1:
                return {
                    "level": "good",
                    "message": "良い感じです！語尾までしっかり発音できています。",
                }
            elif drop_rate < 0.25:
                return {
                    "level": "medium",
                    "message": "語尾がやや弱まっています。少しだけ意識しましょう。",
                }
            else:
                return {
                    "level": "bad",
                    "message": "語尾の音量が大きく低下しています。文末を意識して！",
                }
    except Exception as e:
        return {
            "level": "error",
            "message": "フィードバック生成中にエラーが発生しました。",
        }

def analyze_volume_realtime(audio_segment):
    """リアルタイム音量分析（マイク補正付き）"""
    try:
        extractor = VoiceFeatureExtractor()
        features = extractor.extract_realtime_features(audio_segment)
        evaluation = evaluate_clarity(features)
        return features, evaluation
    except Exception as e:
        default_features = {
            'mean_volume': 0, 'std_volume': 0, 'end_volume': 0,
            'middle_volume': 0, 'end_drop_rate': 0, 'intimacy_level': 'unknown'
        }
        default_evaluation = {
            'clarity_level': '評価不可', 'advice': 'リアルタイム分析でエラーが発生しました。',
            'score': 0, 'avg_drop_rate': 0, 'recording_context': '不明'
        }
        return default_features, default_evaluation
        
