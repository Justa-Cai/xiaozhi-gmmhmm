import os
import numpy as np
import glob
from .feature_extraction import extract_mfcc_from_file

def load_data(data_dir, feature_params):
    """
    加载指定目录下的所有 WAV 文件，提取特征，并按子目录（类别）分组。

    Args:
        data_dir (str): 包含子目录（每个子目录代表一个类别）的数据根目录。
        feature_params (dict): 传递给 extract_mfcc_from_file 的参数字典。
                           例如: {'num_cep': 13, 'frame_len': 0.025, 'frame_stride': 0.01}

    Returns:
        dict: 一个字典，键是类别名称（子目录名），值是该类别所有文件特征的列表。
              例如: {'keyword1': [features1, features2, ...], 'background': [...]}。
        list: 所有特征对应的类别标签列表 (与所有特征文件一一对应，顺序遍历)。
        list: 包含所有文件特征的列表 (顺序遍历)。
        list: 包含所有文件长度（帧数）的列表 (顺序遍历)。
    """
    features_dict = {}
    all_labels = []
    all_features_list = []
    all_lengths = []

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return {}, [], [], []

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        print(f"Warning: No subdirectories found in '{data_dir}'. Expecting subdirectories for classes.")
        return {}, [], [], []

    print(f"Loading data from: {data_dir}")
    print(f"Found categories: {subdirs}")

    for label in subdirs:
        category_dir = os.path.join(data_dir, label)
        features_dict[label] = []
        wav_files = glob.glob(os.path.join(category_dir, '*.wav'))
        print(f"Processing category '{label}': found {len(wav_files)} WAV files.")

        if not wav_files:
            print(f"Warning: No WAV files found in {category_dir}")
            continue

        for wav_path in wav_files:
            features, sample_rate = extract_mfcc_from_file(wav_path, **feature_params)
            if features is not None:
                features_dict[label].append(features)
                all_labels.append(label)
                all_features_list.append(features)
                all_lengths.append(len(features)) # len(features) 是帧数
            else:
                print(f"Skipping file due to extraction error: {wav_path}")

    print(f"Data loading complete. Total files processed successfully: {len(all_features_list)}")
    return features_dict, all_labels, all_features_list, all_lengths

if __name__ == '__main__':
    # 示例用法 (需要你在 gmm_hmm_keyword_spotting 目录下创建 data/test_cat/dummy.wav)
    print("\nRunning utils.py example...")
    example_data_dir = '../data' # 假设脚本在 src/ 目录下运行
    dummy_cat_dir = os.path.join(example_data_dir, 'test_cat')
    dummy_wav_path = os.path.join(dummy_cat_dir, 'dummy_util_test.wav')

    # 创建测试目录和文件
    os.makedirs(dummy_cat_dir, exist_ok=True)
    try:
        from scipy.io import wavfile
        import numpy as np
        sr = 16000
        duration = 1.5
        frequency = 220
        t = np.linspace(0., duration, int(sr * duration), endpoint=False)
        amplitude = 32767 * 0.6
        dummy_signal_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        wavfile.write(dummy_wav_path, sr, dummy_signal_int16)
        print(f"Created dummy file for testing: {dummy_wav_path}")

        # 定义特征提取参数
        f_params = {'num_cep': 13, 'frame_len': 0.025, 'frame_stride': 0.01}

        # 加载数据
        features_d, labels, features_l, lengths = load_data(example_data_dir, f_params)

        # 打印加载结果摘要
        print("\nLoad data results:")
        if features_d:
            print(f"Categories found: {list(features_d.keys())}")
            for cat, feats in features_d.items():
                print(f"  Category '{cat}': {len(feats)} files loaded.")
                if feats:
                    print(f"    Example feature shape for '{cat}': {feats[0].shape}")
            print(f"Total labels collected: {len(labels)}")
            print(f"Total feature arrays collected: {len(features_l)}")
            print(f"Total lengths collected: {len(lengths)}")
            # print(f"Labels: {labels}")
            # print(f"Lengths: {lengths}")
        else:
            print("No data was loaded successfully.")

    except ImportError:
        print("Error: scipy and numpy are required to run the example.")
    except Exception as e:
        print(f"An error occurred during the utils example: {e}")
    finally:
        # 清理测试文件和目录
        if os.path.exists(dummy_wav_path):
            try:
                os.remove(dummy_wav_path)
                print(f"\nRemoved dummy file: {dummy_wav_path}")
            except OSError as e:
                print(f"Error removing dummy file {dummy_wav_path}: {e}")
        # 尝试删除目录，如果它是空的
        if os.path.exists(dummy_cat_dir):
            try:
                os.rmdir(dummy_cat_dir)
                print(f"Removed dummy directory: {dummy_cat_dir}")
            except OSError as e:
                # 如果目录非空或其他错误，则不删除
                # print(f"Could not remove directory {dummy_cat_dir}: {e}")
                pass 