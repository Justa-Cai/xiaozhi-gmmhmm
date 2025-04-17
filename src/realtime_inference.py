import os
import numpy as np
import joblib
import sounddevice as sd
from collections import deque
import time
import sys
import glob

try:
    from .feature_extraction import extract_mfcc
except ImportError:
    # 如果直接运行此脚本，尝试从父目录导入
    from feature_extraction import extract_mfcc


# --- 配置参数 ---
# 使用绝对路径确保文件定位准确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (项目根目录)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')    # 模型加载目录
SCALER_FILENAME = 'scaler.pkl'                 # 标准化器文件名
MODEL_FILENAME_PATTERN = 'gmmhmm_*.pkl'          # 模型文件名模式

# MFCC 特征参数 (必须与训练时完全一致)
FEATURE_PARAMS = {
    'num_cep': 13,
    'frame_len': 0.025,
    'frame_stride': 0.01,
    'preemph': 0.97
}

# 实时音频流参数
SAMPLE_RATE = 16000        # 采样率 (Hz) - 必须与训练数据一致
CHUNK_DURATION_MS = 100    # 每次读取的音频块时长 (毫秒)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) # 每块的采样点数
# 缓冲区需要足够长以计算至少一帧特征，考虑帧长和步长
# 至少需要 frame_len 长度的数据
# 为了平滑，可以使用更长的缓冲区，例如几百毫秒
BUFFER_DURATION_S = 0.5  # 用于特征提取的音频缓冲区时长 (秒) - 可调整
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION_S)

# 检测参数 (需要根据实际效果仔细调整)
DETECTION_THRESHOLD_OFFSET = 300  # 关键词得分需要比背景得分高出多少 (log-likelihood difference) - 可调整
SMOOTHING_WINDOW_SIZE = 8     # 检测结果平滑窗口大小 (多少个有效检测帧)
MIN_DETECTION_COUNT = 3       # 平滑窗口中至少需要多少帧检测为同一个词才确认
SILENCE_THRESHOLD = 0.005     # RMS 能量阈值，低于此值认为是静音 (归一化后 [-1, 1]) - 可调整
BACKGROUND_LABEL = 'background' # 背景/噪音模型的标签名 (需要与训练时的目录名一致)
DEBOUNCE_TIME_S = 1.0         # 检测到一次后，至少等待这么久才报下一次

# ----------------

# 全局变量用于 callback 和主线程通信
audio_buffer = deque(maxlen=BUFFER_SAMPLES)
detection_queue = deque(maxlen=SMOOTHING_WINDOW_SIZE)
last_detection_time = 0
models_global = None
scaler_global = None
keyword_labels_global = []
background_model_global = None

def load_models_and_scaler(model_dir):
    """加载所有 GMM-HMM 模型和标准化器"""
    models = {}
    scaler = None
    keyword_labels = []
    background_model = None

    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        return None, None, [], None

    scaler_path = os.path.join(model_dir, SCALER_FILENAME)
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded StandardScaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
            return None, None, [], None
    else:
        print(f"Error: StandardScaler file '{SCALER_FILENAME}' not found in '{model_dir}'")
        return None, None, [], None

    model_files = glob.glob(os.path.join(model_dir, MODEL_FILENAME_PATTERN))
    if not model_files:
        print(f"Error: No model files found in '{model_dir}' matching '{MODEL_FILENAME_PATTERN}'")
        return None, scaler, [], None

    print(f"Found model files: {model_files}")
    loaded_labels = set()
    for model_path in model_files:
        try:
            model = joblib.load(model_path)
            label = os.path.basename(model_path).replace('gmmhmm_', '').replace('.pkl', '')
            if not label:
                print(f"Warning: Could not extract label from filename '{model_path}'. Skipping.")
                continue

            models[label] = model
            loaded_labels.add(label)
            print(f"Loaded model for '{label}' from {model_path}")

            if label == BACKGROUND_LABEL:
                background_model = model
            else:
                keyword_labels.append(label)

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    if not models:
        print("Error: No models were successfully loaded.")
        return None, scaler, [], None

    if background_model is None and BACKGROUND_LABEL in loaded_labels:
         print(f"Warning: Background label '{BACKGROUND_LABEL}' found but model failed/not assigned.")
    elif background_model is None:
         print(f"Warning: Background model '{BACKGROUND_LABEL}' not found. Detection might be less reliable.")

    if not keyword_labels:
        print(f"Warning: No keyword models loaded (excluding '{BACKGROUND_LABEL}').")

    return models, scaler, keyword_labels, background_model


def audio_callback(indata, frames, time_info, status):
    """sounddevice 回调函数，处理输入的音频数据"""
    global audio_buffer, detection_queue, last_detection_time, models_global, scaler_global, keyword_labels_global, background_model_global

    if status:
        print(f"Status: {status}", file=sys.stderr)

    # 确保数据是 float32 类型
    audio_chunk = indata[:, 0].astype(np.float32) # 取第一个声道并转为 float32

    # --- VAD (基于能量) ---
    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < SILENCE_THRESHOLD:
        # print(".", end="", flush=True) # 可以取消注释来观察静音检测
        # 静音时可以考虑清空 detection_queue 或减少计算
        # detection_queue.clear() # 如果希望静音打断连续检测
        return # 暂时简单处理：静音时不处理

    # 将新数据块添加到缓冲区
    audio_buffer.extend(audio_chunk)

    # 确保缓冲区中有足够的数据进行特征提取
    if len(audio_buffer) < int(FEATURE_PARAMS['frame_len'] * SAMPLE_RATE):
        return # 数据不足一帧，等待更多数据

    # --- 特征提取与评分 ---
    signal_to_process = np.array(audio_buffer)
    try:
        features = extract_mfcc(signal_to_process, SAMPLE_RATE, **FEATURE_PARAMS)
        if features is None or len(features) == 0:
            # print("W: Feature extraction failed or yielded no frames.")
            return
        scaled_features = scaler_global.transform(features)

        # 计算所有模型的得分 (log likelihood)
        scores = {}
        max_keyword_score = -np.inf
        detected_keyword = None

        for label, model in models_global.items():
            try:
                # model.score 需要 2D array
                scores[label] = model.score(scaled_features)
            except Exception as e:
                # print(f"E: Scoring failed for '{label}': {e}")
                scores[label] = -np.inf # 出错时给最低分

            if label != BACKGROUND_LABEL and scores[label] > max_keyword_score:
                max_keyword_score = scores[label]

        # --- 检测逻辑 ---
        background_score = scores.get(BACKGROUND_LABEL, -np.inf) # 获取背景得分，若无则为负无穷

        # 检查最佳关键词得分是否足够高于背景得分
        if max_keyword_score > background_score + DETECTION_THRESHOLD_OFFSET:
            print(f"max_keyword_score-background_score: {max_keyword_score-background_score}")
            # 找出得分最高的关键词
            for label in keyword_labels_global:
                if scores[label] == max_keyword_score:
                    detected_keyword = label
                    break
        else:
            detected_keyword = None # 没有关键词显著胜出

        detection_queue.append(detected_keyword) # 将当前帧的检测结果（或None）加入队列

        # --- 平滑与触发 ---
        # 统计当前平滑窗口内各关键词出现的次数
        counts = {}
        valid_detections = 0
        for item in detection_queue:
            if item is not None:
                counts[item] = counts.get(item, 0) + 1
                valid_detections += 1

        final_detection = None
        if valid_detections >= MIN_DETECTION_COUNT: # 确保窗口内有足够多的非None检测结果
            for keyword, count in counts.items():
                if count >= MIN_DETECTION_COUNT: # 如果某个词出现次数达到阈值
                    final_detection = keyword
                    break # 只取第一个满足条件的

        # --- 触发与防抖 ---
        current_time = time.time()
        if final_detection is not None and (current_time - last_detection_time > DEBOUNCE_TIME_S):
            print(f"\n>>> Detected Keyword: [ {final_detection.upper()} ] <<< ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
            last_detection_time = current_time
            detection_queue.clear() # 检测到后清空队列，避免连续触发

    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        # 发生错误时可以考虑清空 buffer 或 queue
        # audio_buffer.clear()
        # detection_queue.clear()


def real_time_inference():
    """执行实时推理主逻辑"""
    global models_global, scaler_global, keyword_labels_global, background_model_global, audio_buffer

    print(f"Loading models from: {MODEL_DIR}")
    models, scaler, keyword_labels, background_model = load_models_and_scaler(MODEL_DIR)

    if models is None or scaler is None:
        print("\nFailed to load models or scaler. Please ensure training was successful.")
        print(f"Check directory: {os.path.abspath(MODEL_DIR)}")
        return

    if not keyword_labels:
        print("\nWarning: No keyword models loaded. Inference will not detect any keywords.")
        # 可以选择退出或继续（只运行背景模型等）
        # return

    models_global = models
    scaler_global = scaler
    keyword_labels_global = keyword_labels
    background_model_global = background_model

    # 初始化音频缓冲区
    audio_buffer.clear()
    detection_queue.clear()

    print("\nStarting real-time inference... Press Ctrl+C to stop.")
    print(f"- Sample Rate: {SAMPLE_RATE} Hz")
    print(f"- Chunk Size: {CHUNK_SAMPLES} samples ({CHUNK_DURATION_MS} ms)")
    print(f"- Processing Buffer: {BUFFER_SAMPLES} samples ({BUFFER_DURATION_S:.2f} s)")
    print(f"- Keyword Models: {keyword_labels if keyword_labels else 'None'}")
    print(f"- Background Model: {'Yes' if background_model else 'No'} (Label: '{BACKGROUND_LABEL}')")
    print(f"- Detection Threshold Offset: {DETECTION_THRESHOLD_OFFSET:.2f}")
    print(f"- Smoothing Window: {SMOOTHING_WINDOW_SIZE}, Min Count: {MIN_DETECTION_COUNT}")
    print(f"- Silence Threshold (RMS): {SILENCE_THRESHOLD:.4f}")
    print(f"- Debounce Time: {DEBOUNCE_TIME_S:.1f} s")
    print("-" * 30)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype='float32', # 使用 float32 以便 VAD 和处理
                            blocksize=CHUNK_SAMPLES, # 每次回调处理的块大小
                            callback=audio_callback):
            print("Microphone stream started. Listening...")
            while True:
                # 主线程保持运行，回调函数在后台处理音频
                time.sleep(0.1)
                print(".", end="", flush=True) # 在主线程打印点表示程序仍在运行

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
    except Exception as e:
        # 尝试报告更具体的 sounddevice 错误
        if isinstance(e, sd.PortAudioError):
            print(f"\nSounddevice Error: {e}")
            print("Please check your audio device configuration and permissions.")
            print("Available devices:")
            try:
                print(sd.query_devices())
            except Exception as qe:
                print(f"Could not query audio devices: {qe}")
        else:
            print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("Audio stream stopped.")


if __name__ == '__main__':
    real_time_inference() 