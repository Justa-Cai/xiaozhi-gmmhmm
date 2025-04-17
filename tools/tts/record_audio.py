import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import argparse
import time

# --- 配置 ---
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1  # 单声道
DEFAULT_DTYPE = 'int16' # 16-bit
# DEFAULT_OUTPUT_DIR = "data" # 不再需要默认输出目录

def record_audio(duration, output_path_arg):
    """
    录制指定时长的音频并保存为 WAV 文件。

    Args:
        duration (int): 录音时长（秒）。
        output_path_arg (str): 用户通过命令行指定的输出路径参数。
    """
    sample_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    dtype = DEFAULT_DTYPE
    # output_dir = DEFAULT_OUTPUT_DIR # 移除

    # --- 处理输出路径 ---
    output_path = output_path_arg
    # 检查用户是否提供了一个目录（以 / 或 \ 结尾）
    if output_path.endswith(os.path.sep):
        # 用户提供的是目录，生成默认文件名
        default_filename = f"recording_{int(time.time())}.wav"
        output_path = os.path.join(output_path, default_filename)
        print(f"检测到输出路径为目录，将使用默认文件名: {default_filename}")
    # else: 用户提供了完整的文件路径

    # 确保基础输出目录存在 (在 join 之前创建 output_dir 变量不再适用)
    # os.makedirs(output_dir, exist_ok=True) # 移除

    # output_path = os.path.join(output_dir, output_filename) # 直接使用处理后的 output_path

    print(f"开始录音，时长: {duration} 秒...")
    print(f"采样率: {sample_rate} Hz")
    print(f"通道数: {channels}")
    print(f"位深度: {dtype}")
    print(f"保存路径: {output_path}")

    # 录音
    recording = None # 初始化 recording 变量
    try:
        # 使用 Stream 方式录音，以便我们可以控制停止
        stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype)
        stream.start()
        print("按 Ctrl+C 停止录音...")
        frames = []
        # 持续录音，直到按下 Ctrl+C 或达到指定时长（如果需要限制时长，可以添加计时逻辑）
        # 这里我们让它一直录音，直到 Ctrl+C
        # while True: # 如果需要无限时长录音，则取消注释此行并删除下面几行
        #     frames.append(stream.read(sample_rate)[0]) # 每次读取1秒的数据

        # 或者，按原有时长逻辑，但允许中途Ctrl+C
        num_frames_total = int(duration * sample_rate)
        frames_recorded = 0
        block_size = 1024 # 每次读取的块大小，可以调整
        while frames_recorded < num_frames_total:
             read_frames = min(block_size, num_frames_total - frames_recorded)
             frame_data, overflowed = stream.read(read_frames)
             if overflowed:
                 print("警告：录音缓冲区溢出！")
             frames.append(frame_data)
             frames_recorded += len(frame_data)

        # recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=dtype)
        # sd.wait()  # 等待录音完成

    except KeyboardInterrupt:
        print("\n录音被中断 (Ctrl+C)。")
    except Exception as e:
        print(f"录音时发生错误: {e}")
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
            print("录音流已停止。")
            if frames:
                recording = np.concatenate(frames, axis=0) # 将所有帧合并

    # print("录音结束。") # 这行现在可能在finally之前或中断时打印，调整位置或删除

    # 保存为 WAV 文件
    if recording is not None and recording.size > 0: # 确保有录音数据才保存
        print("正在保存文件...")
        try:
            # 确保最终输出文件的目录存在
            output_dir_full = os.path.dirname(output_path)
            if output_dir_full: # 避免在 output_path 没有目录部分时尝试创建 ''
                os.makedirs(output_dir_full, exist_ok=True)

            write(output_path, sample_rate, recording)
            print(f"音频已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存 WAV 文件时发生错误: {e}")
    elif recording is not None:
         print("没有录制到有效音频数据，不保存文件。")
    else:
        # 如果录音开始前就出错，recording 可能为 None
        print("未能开始录音，无法保存文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="录制音频并保存为 WAV 文件。")
    parser.add_argument("-d", "--duration", type=int, default=5, help="录音时长（秒），默认为 5 秒。")
    parser.add_argument("-o", "--output", type=str, default=f"recording_{int(time.time())}.wav",
                        help="输出 WAV 文件路径。如果以 '/' 或 '\' 结尾，则视为目录，并在此目录下生成默认文件名。默认为 'recording_时间戳.wav'。")

    args = parser.parse_args()

    record_audio(args.duration, args.output) 