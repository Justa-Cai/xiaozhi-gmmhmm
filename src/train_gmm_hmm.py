import os
import numpy as np
import joblib
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from .utils import load_data

# --- 配置参数 ---
DATA_DIR = 'data'       # 训练数据根目录 (相对于 src/)
MODEL_DIR = 'models'    # 模型保存目录 (相对于 src/)
MODEL_FILENAME_TEMPLATE = 'gmmhmm_{label}.pkl' # 模型文件名模板
SCALER_FILENAME = 'scaler.pkl' # 标准化器文件名

# MFCC 特征参数 (需要与 feature_extraction.py 和 realtime_inference.py 中保持一致)
FEATURE_PARAMS = {
    'num_cep': 13,
    'frame_len': 0.025,
    'frame_stride': 0.01,
    'preemph': 0.97
}

# HMM 模型参数 (需要根据实际情况调整)
HMM_PARAMS = {
    'n_components': 5,   # HMM 状态数 (经验值，通常与音素或音节有关，可调整)
    'n_mix': 3,        # 每个状态的 GMM 混合数 (经验值，可调整)
    'covariance_type': 'diag', # GMM 协方差类型 ('diag' 常用且计算较快)
    'n_iter': 100,      # HMM 训练迭代次数
    'tol': 1e-3,       # 收敛阈值
    'verbose': True,     # 是否打印训练过程信息
    'init_params': 'stmcw', # 初始化参数: startprob, transmat, means, covars, weights
    'params': 'stmcw',     # 训练时调整的参数
    'covars_prior': 1e-2, 
}

# ----------------

def train_models(data_dir, model_dir, feature_params, hmm_params):
    """
    加载数据，训练 GMM-HMM 模型并保存。
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. 加载数据和提取特征
    print("Loading data and extracting features...")
    features_dict, _, all_features_list, all_lengths = load_data(data_dir, feature_params)

    if not all_features_list:
        print("Error: No features were extracted. Cannot train models.")
        print(f"Please check if '{data_dir}' contains valid WAV files in subdirectories.")
        return

    # 2. 数据标准化 (非常重要)
    # 将所有特征数据合并用于计算均值和方差
    print("Fitting StandardScaler...")
    all_features_concat = np.concatenate(all_features_list)
    scaler = StandardScaler()
    scaler.fit(all_features_concat)
    print(f"Scaler mean shape: {scaler.mean_.shape}, variance shape: {scaler.var_.shape}")

    # 保存标准化器
    scaler_path = os.path.join(model_dir, SCALER_FILENAME)
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to {scaler_path}")

    # 对每个文件的特征进行标准化
    print("Applying scaling to features...")
    scaled_features_dict = {}
    scaled_features_list = []
    for label, features_list in features_dict.items():
        scaled_features_dict[label] = []
        for features in features_list:
            scaled_features = scaler.transform(features)
            scaled_features_dict[label].append(scaled_features)
            scaled_features_list.append(scaled_features) # 更新 all_features_list 为 scaled 版本

    # 3. 为每个类别训练一个 GMM-HMM 模型
    print("\nTraining GMM-HMM models...")
    models = {}
    for label, features_list in scaled_features_dict.items():
        if not features_list:
            print(f"Skipping category '{label}': No features available.")
            continue

        print(f"-- Training model for category: '{label}' --")
        # hmmlearn 需要将每个文件的特征作为一个独立的序列传入
        # 同时需要提供每个序列的长度
        label_features_concat = np.concatenate(features_list)
        label_lengths = [len(f) for f in features_list]

        if label_features_concat.shape[0] < hmm_params['n_components']:
            print(f"Warning: Not enough data for category '{label}' to train HMM with {hmm_params['n_components']} states. Skipping.")
            continue
        if not label_lengths:
             print(f"Warning: Zero length sequence found for category '{label}'. Skipping.")
             continue

        model = hmm.GMMHMM(**hmm_params)

        try:
            # 确保没有 NaN 或 Inf
            if np.any(np.isnan(label_features_concat)) or np.any(np.isinf(label_features_concat)):
                print(f"Error: NaN or Inf found in features for category '{label}'. Skipping training.")
                continue

            model.fit(label_features_concat, lengths=label_lengths)
            models[label] = model
            print(f"Model training for '{label}' finished. Converged: {model.monitor_.converged}")

            # 保存模型
            model_filename = MODEL_FILENAME_TEMPLATE.format(label=label)
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"Model for '{label}' saved to {model_path}")

        except ValueError as ve:
             print(f"Error training model for '{label}': {ve}")
             print("This might be due to insufficient data, all-zero features, or issues with parameters.")
             print(f"Feature shape: {label_features_concat.shape}, Lengths: {label_lengths[:5]}...")
             continue
        except Exception as e:
            print(f"An unexpected error occurred during training for '{label}': {e}")
            continue

    print("\nAll models trained and saved.")
    return models

if __name__ == '__main__':
    print("Starting GMM-HMM training process...")
    # 确保模型目录存在
    if not os.path.exists(MODEL_DIR):
        print(f"Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)

    # 检查数据目录是否存在
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please create it and populate it with subdirectories containing WAV files for each keyword/background.")
         print("Example structure:")
         print("../data/")
         print("  ├── keyword1/")
         print("  │   ├── file1.wav")
         print("  │   └── file2.wav")
         print("  └── background/")
         print("      └── noise1.wav")
    else:
        train_models(DATA_DIR, MODEL_DIR, FEATURE_PARAMS, HMM_PARAMS) 