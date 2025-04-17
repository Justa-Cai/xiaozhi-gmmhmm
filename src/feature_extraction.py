import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
# from scipy.signal import lfilter, hamming # lfilter 和 hamming 似乎没有直接使用，暂时注释
import os

def extract_mfcc(audio_signal, sample_rate, num_cep=13, frame_len=0.025, frame_stride=0.01, preemph=0.97):
    """
    从音频信号中提取 MFCC 特征。

    Args:
        audio_signal (np.ndarray): 输入的音频时间序列信号。
        sample_rate (int): 音频的采样率。
        num_cep (int): 返回的倒谱系数数量 (包括能量)。
        frame_len (float): 帧长度 (秒)。
        frame_stride (float): 帧步长 (秒)。
        preemph (float): 预加重系数。

    Returns:
        np.ndarray: MFCC 特征矩阵 (num_frames x num_cep)。
    """
    # 预加重
    # Note: python_speech_features.mfcc 内部可以处理预加重，但我们这里手动做一次作为示例
    # 如果 preemph 设置为 0，则 mfcc 函数内部的预加重会被跳过
    emphasized_signal = np.append(audio_signal[0], audio_signal[1:] - preemph * audio_signal[:-1])

    # 计算 MFCC 特征
    features = mfcc(emphasized_signal,
                      samplerate=sample_rate,
                      winlen=frame_len,
                      winstep=frame_stride,
                      numcep=num_cep,
                      nfilt=26,          # 滤波器组的数量 (常用值)
                      nfft=512,          # FFT 点数 (常用值，通常是2的幂次)
                      lowfreq=0,         # 最低频率
                      highfreq=None,     # 最高频率 (None 表示 sample_rate / 2)
                      preemph=0,         # 我们已经在外部做了预加重，这里设为0
                      ceplifter=22,      # 倒谱提升窗口大小 (常用值)
                      appendEnergy=True, # 添加能量作为第0个倒谱系数
                      winfunc=np.hamming # 窗函数 (hamming 窗)
                      )
    return features

def extract_mfcc_from_file(wav_path, **kwargs):
    """
    从 WAV 文件中读取音频并提取 MFCC 特征。

    Args:
        wav_path (str): WAV 文件的路径。
        **kwargs: 传递给 extract_mfcc 的其他参数。

    Returns:
        np.ndarray: MFCC 特征矩阵，如果读取失败则返回 None。
        int: 音频采样率，如果读取失败则返回 None。
    """
    try:
        sample_rate, signal = wavfile.read(wav_path)

        # 确保是单声道
        if signal.ndim > 1:
            # 取第一个声道
            signal = signal[:, 0]
            # print(f"Warning: Audio file {wav_path} is not mono. Taking the first channel.")

        # 归一化信号到 [-1, 1] 范围 (float32)
        # 这是许多信号处理库的常见做法，包括 python_speech_features
        if np.issubdtype(signal.dtype, np.integer):
             # 对于整数类型，除以最大可能值
             max_val = np.iinfo(signal.dtype).max
             signal = signal.astype(np.float32) / max_val
        elif np.issubdtype(signal.dtype, np.floating):
             # 如果已经是浮点数，确保是 float32
             signal = signal.astype(np.float32)
             # 检查范围是否大致在 [-1, 1] 内，如果不是可能需要调整
             # if np.max(np.abs(signal)) > 1.0:
             #    print(f"Warning: Floating point signal in {wav_path} seems not normalized.")
        else:
             print(f"Warning: Unsupported audio format {signal.dtype} in {wav_path}. Skipping.")
             return None, None

        # 提取特征
        features = extract_mfcc(signal, sample_rate, **kwargs)
        return features, sample_rate
    except FileNotFoundError:
        print(f"Error: File not found at {wav_path}")
        return None, None
    except ValueError as ve:
        print(f"Error processing file {wav_path}: {ve}. Check if it's a valid WAV file.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred processing file {wav_path}: {e}")
        return None, None

if __name__ == '__main__':
    # 示例：提取一个假想的 WAV 文件的 MFCC
    dummy_wav_path = 'dummy_audio_mfcc_test.wav'
    # 创建一个虚拟的WAV文件用于测试
    sr = 16000
    duration = 1 # seconds
    frequency = 440 # Hz

    # 创建虚拟 WAV 文件
    try:
        t = np.linspace(0., duration, int(sr * duration), endpoint=False)
        amplitude = 32767 * 0.5 # 峰值幅度为 int16 最大值的一半
        dummy_signal_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        wavfile.write(dummy_wav_path, sr, dummy_signal_int16)
        print(f"Created dummy audio file: {dummy_wav_path}")

        # 测试特征提取
        print("\nTesting feature extraction...")
        features, rate = extract_mfcc_from_file(dummy_wav_path, num_cep=13, frame_len=0.025, frame_stride=0.01)

        if features is not None:
            print(f"Successfully extracted features from {dummy_wav_path}")
            print(f"Shape: {features.shape}")
            print(f"Sample rate: {rate}")
            # print("First 2 frames:\n", features[:2]) # Corrected comment
        else:
            print(f"Failed to extract features from {dummy_wav_path}.")

    except ImportError:
         print("Error: scipy.io.wavfile required for the example. Please install scipy.")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
    finally:
        # 清理虚拟文件，无论成功还是失败
        if os.path.exists(dummy_wav_path):
            try:
                os.remove(dummy_wav_path)
                print(f"\nRemoved dummy audio file: {dummy_wav_path}")
            except OSError as e:
                print(f"Error removing dummy file {dummy_wav_path}: {e}") 