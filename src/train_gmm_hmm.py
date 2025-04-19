# -*- coding: utf-8 -*-
import os
import numpy as np
import joblib
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import struct # <--- 添加 struct 模块
import sys # <-- Import sys earlier

# 确保 utils 可以被导入
try:
    from .utils import load_data
except ImportError:
    # 如果直接运行此脚本，尝试从父目录导入
    SCRIPT_DIR_FOR_IMPORT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_IMPORT = os.path.dirname(SCRIPT_DIR_FOR_IMPORT)
    # Add project root to path to allow absolute import 'from src.utils ...'
    if PROJECT_ROOT_FOR_IMPORT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_IMPORT)
    try:
        from src.utils import load_data
    except ImportError as e:
        print(f"Failed to import 'load_data' even after adjusting path: {e}")
        # Provide more context for the user
        print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT_FOR_IMPORT}")
        print(f"Attempted import: from src.utils import load_data")
        print("Please ensure 'utils.py' exists within the 'src' directory.")
        sys.exit(1) # Exit if import fails


# --- 配置参数 ---
# 使用绝对路径可能更可靠，特别是当从不同位置运行时
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # 项目根目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')       # 训练数据根目录
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')    # 模型保存目录
MODEL_FILENAME_TEMPLATE = 'gmmhmm_{label}.pkl' # 模型文件名模板
SCALER_FILENAME = 'scaler.pkl' # 标准化器文件名
# 新增：二进制文件名模板
MODEL_BIN_FILENAME_TEMPLATE = 'gmmhmm_{label}.bin'
SCALER_BIN_FILENAME = 'scaler.bin'

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
    # 'covars_prior': 1e-2, # GMMHMM 不直接接受 covars_prior，通过 min_covar 控制
    'min_covar': 1e-3,      # 防止协方差过小导致数值问题
}

# 防止 log(0) 的小常数
LOG_EPS = 1e-20

# ----------------

# --- 新增：导出函数 ---

def export_scaler_to_bin(scaler, filepath):
    """将 StandardScaler 对象导出为 C 可读的二进制格式"""
    print(f"Exporting scaler to {filepath}...")
    try:
        feature_dim = scaler.mean_.shape[0]
        mean = scaler.mean_.astype(np.float32)

        # 计算标准差倒数 (C 代码需要)
        # scaler.scale_ 存储的是标准差
        std_dev = scaler.scale_.astype(np.float32)
        # 处理标准差接近零的情况，防止除零
        std_dev[std_dev < LOG_EPS] = LOG_EPS
        inv_std_dev = 1.0 / std_dev

        with open(filepath, 'wb') as f:
            # 1. 写入特征维度 (int)
            f.write(struct.pack('i', feature_dim))
            # 2. 写入均值向量 (float array)
            f.write(struct.pack(f'{feature_dim}f', *mean))
            # 3. 写入标准差倒数向量 (float array)
            f.write(struct.pack(f'{feature_dim}f', *inv_std_dev))
        print(f"Scaler exported successfully. Feature dim: {feature_dim}")
        return True
    except Exception as e:
        print(f"Error exporting scaler: {e}")
        return False

def export_hmm_to_bin(model, label, filepath):
    """将 GMMHMM 对象导出为 C 可读的二进制格式"""
    print(f"Exporting HMM model for '{label}' to {filepath}...")
    try:
        num_states = model.n_components
        num_features = model.n_features
        num_mix = model.n_mix # GMM 混合数

        with open(filepath, 'wb') as f:
            # 1. 写入标签 (int length, char* label)
            label_bytes = label.encode('utf-8')
            label_len = len(label_bytes)
            f.write(struct.pack('i', label_len))
            f.write(struct.pack(f'{label_len}s', label_bytes))

            # 2. 写入 HMM 参数 (int num_states, int num_features)
            f.write(struct.pack('i', num_states))
            f.write(struct.pack('i', num_features))

            # 3. 写入对数初始概率 (float array)
            log_start_prob = np.log(model.startprob_ + LOG_EPS).astype(np.float32)
            f.write(struct.pack(f'{num_states}f', *log_start_prob))

            # 4. 写入对数转移概率 (float matrix, flattened row-major)
            log_trans_mat = np.log(model.transmat_ + LOG_EPS).astype(np.float32)
            f.write(struct.pack(f'{num_states * num_states}f', *log_trans_mat.ravel()))

            # 5. 写入每个状态的 GMM 参数
            for i in range(num_states):
                # a. 写入 GMM 分量数 (int) - 注意：hmmlearn 的 GMMHMM n_mix 对所有状态相同
                f.write(struct.pack('i', num_mix))

                # b. 写入 GMM 对数权重 (float array)
                log_weights = np.log(model.weights_[i] + LOG_EPS).astype(np.float32)
                f.write(struct.pack(f'{num_mix}f', *log_weights))

                # c. 写入 GMM 均值 (float matrix, flattened row-major)
                means = model.means_[i].astype(np.float32) # shape (n_mix, n_features)
                f.write(struct.pack(f'{num_mix * num_features}f', *means.ravel()))

                # d. 写入 GMM 方差倒数 (float matrix, flattened row-major)
                # model.covars_ 是方差 (因为 covariance_type='diag')
                covars = model.covars_[i].astype(np.float32) # shape (n_mix, n_features)
                # 处理方差接近零的情况
                covars[covars < LOG_EPS] = LOG_EPS
                inv_vars = 1.0 / covars
                f.write(struct.pack(f'{num_mix * num_features}f', *inv_vars.ravel()))

                # e. 写入 GMM 对数行列式相关项 (float array)
                # Precompute -0.5 * (sum(log(var_d)) + D*log(2*pi)) for each component k
                log_dets = np.zeros(num_mix, dtype=np.float32)
                log_2pi = np.log(2 * np.pi).astype(np.float32)
                for k in range(num_mix):
                    log_var_sum = np.sum(np.log(covars[k] + LOG_EPS)) # Use covars here
                    log_dets[k] = -0.5 * (log_var_sum + num_features * log_2pi)
                f.write(struct.pack(f'{num_mix}f', *log_dets))

        print(f"Model '{label}' exported successfully.")
        return True
    except AttributeError as ae:
         print(f"Error exporting HMM model for '{label}': Missing attribute {ae}. Model might not be fully trained or has unexpected structure.")
         # 打印模型的部分属性以帮助调试
         print(f"  Model type: {type(model)}")
         print(f"  Attributes: {dir(model)}")
         return False
    except Exception as e:
        print(f"Error exporting HMM model for '{label}': {e}")
        return False

# ----------------

def train_models(data_dir, model_dir, feature_params, hmm_params):
    """
    加载数据，训练 GMM-HMM 模型并保存 .pkl 和 .bin 文件。
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. 加载数据和提取特征
    print("Loading data and extracting features...")
    # 使用绝对路径加载数据
    features_dict, _, all_features_list, all_lengths = load_data(data_dir, feature_params)

    if not all_features_list:
        print(f"Error: No features were extracted from '{data_dir}'. Cannot train models.")
        print("Please check if the directory contains valid WAV files in subdirectories.")
        return

    # 2. 数据标准化 (非常重要)
    print("Fitting StandardScaler...")
    all_features_concat = np.concatenate(all_features_list)
    if np.any(np.isnan(all_features_concat)) or np.any(np.isinf(all_features_concat)):
        print("Error: NaN or Inf found in raw features. Check your audio files or feature extraction.")
        # 定位问题特征文件可能需要更复杂的逻辑
        problematic_indices = np.where(np.isnan(all_features_concat) | np.isinf(all_features_concat))
        print(f"Problematic indices (flattened): {problematic_indices[0][:10]}...") # Show first few
        return

    scaler = StandardScaler()
    try:
        scaler.fit(all_features_concat)
        print(f"Scaler mean shape: {scaler.mean_.shape}, scale (std dev) shape: {scaler.scale_.shape}")
    except ValueError as e:
        print(f"Error fitting scaler: {e}")
        print("This might happen if variance is zero for some features.")
        variances = np.var(all_features_concat, axis=0)
        zero_var_indices = np.where(variances < 1e-10)[0]
        if len(zero_var_indices) > 0:
            print(f"Features with near-zero variance found at indices: {zero_var_indices}")
        return

    # 保存标准化器 (.pkl)
    scaler_path = os.path.join(model_dir, SCALER_FILENAME)
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to {scaler_path}")

    # --- 导出标准化器 (.bin) ---
    scaler_bin_path = os.path.join(model_dir, SCALER_BIN_FILENAME)
    export_scaler_to_bin(scaler, scaler_bin_path)
    # --------------------------

    # 对每个文件的特征进行标准化
    print("Applying scaling to features...")
    scaled_features_dict = {}
    for label, features_list in features_dict.items():
        scaled_features_dict[label] = []
        for features in features_list:
            # 检查原始特征是否有 NaN/Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                 print(f"Warning: NaN/Inf found in features for an item in '{label}' BEFORE scaling. Skipping this item.")
                 continue # 跳过这个有问题的特征文件
            try:
                scaled_features = scaler.transform(features)
                # 再次检查缩放后是否有 NaN/Inf (理论上不应发生，除非 scaler 有问题)
                if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
                    print(f"Warning: NaN/Inf found in features for an item in '{label}' AFTER scaling. Skipping this item.")
                    continue
                scaled_features_dict[label].append(scaled_features)
            except Exception as e:
                print(f"Error scaling features for an item in '{label}': {e}. Skipping this item.")
                continue


    # 3. 为每个类别训练一个 GMM-HMM 模型
    print("\nTraining GMM-HMM models...")
    models = {}
    export_success_count = 0
    export_failure_count = 0
    for label, features_list in scaled_features_dict.items():
        # 过滤掉空的特征列表 (如果在缩放时被跳过)
        valid_features_list = [f for f in features_list if f is not None and len(f) > 0]
        if not valid_features_list:
            print(f"Skipping category '{label}': No valid scaled features available after filtering.")
            continue

        print(f"-- Training model for category: '{label}' --")
        # 使用有效特征进行连接和计算长度
        label_features_concat = np.concatenate(valid_features_list)
        label_lengths = [len(f) for f in valid_features_list]

        # 检查数据量是否足够
        if label_features_concat.shape[0] < hmm_params['n_components']:
            print(f"Warning: Not enough data ({label_features_concat.shape[0]} frames) for category '{label}' "
                  f"to train HMM with {hmm_params['n_components']} states. Skipping.")
            continue
        if not label_lengths:
             print(f"Warning: Zero length sequences found for category '{label}' after filtering. Skipping.")
             continue
        if sum(label_lengths) == 0:
             print(f"Warning: Total length of sequences is zero for category '{label}'. Skipping.")
             continue


        # 检查 HMM 参数兼容性 (特别是 n_mix) - REMOVED version check
        current_hmm_params = hmm_params.copy()
        # if 'n_mix' not in hmm.GMMHMM()._get_init_params_dict(): # 检查是否支持 n_mix
        #      print(f"Warning: hmmlearn version might not support 'n_mix' directly in GMMHMM constructor. Using default mixture components per state.")
             # 如果版本不支持，可能需要移除 n_mix 或采用其他初始化方式
             # current_hmm_params.pop('n_mix', None) # 移除 n_mix 参数
        # 检查 covars_prior 是否在参数中 (新版本 hmmlearn 可能没有)
        # if 'covars_prior' in current_hmm_params and 'covars_prior' not in hmm.GMMHMM()._get_init_params_dict():
        #      print(f"Warning: 'covars_prior' is not a direct parameter in this hmmlearn version. Use 'min_covar' instead.")
        #      current_hmm_params.pop('covars_prior', None) # 移除


        model = hmm.GMMHMM(**current_hmm_params) # Use the prepared params directly

        try:
            # 最终检查 NaN/Inf
            if np.any(np.isnan(label_features_concat)) or np.any(np.isinf(label_features_concat)):
                print(f"Error: NaN or Inf found in concatenated features for '{label}' just before training. Skipping.")
                continue

            model.fit(label_features_concat, lengths=label_lengths)
            models[label] = model
            print(f"Model training for '{label}' finished. Converged: {model.monitor_.converged}")

            # 保存模型 (.pkl)
            model_filename = MODEL_FILENAME_TEMPLATE.format(label=label)
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"Model for '{label}' saved to {model_path}")

            # --- 导出模型 (.bin) ---
            model_bin_filename = MODEL_BIN_FILENAME_TEMPLATE.format(label=label)
            model_bin_path = os.path.join(model_dir, model_bin_filename)
            # 确保模型有必要的属性才尝试导出
            if hasattr(model, 'n_components') and \
               hasattr(model, 'n_features') and \
               hasattr(model, 'n_mix') and \
               hasattr(model, 'startprob_') and \
               hasattr(model, 'transmat_') and \
               hasattr(model, 'weights_') and \
               hasattr(model, 'means_') and \
               hasattr(model, 'covars_'):
                if export_hmm_to_bin(model, label, model_bin_path):
                    export_success_count += 1
                else:
                    export_failure_count += 1
            else:
                 print(f"Warning: Model for '{label}' seems incomplete or lacks expected attributes. Skipping binary export.")
                 print(f"  Attributes found: {dir(model)}")
                 export_failure_count += 1
            # -----------------------

        except ValueError as ve:
             print(f"Error training model for '{label}': {ve}")
             print("This might be due to insufficient data, all-zero features, parameter issues, or NaNs.")
             print(f"Feature shape: {label_features_concat.shape}, Lengths sum: {sum(label_lengths)}")
             # Check first few features for obvious issues
             if len(label_features_concat) > 0:
                 print(f"First feature vector sample: {label_features_concat[0, :5]}...")
             continue
        except Exception as e:
            print(f"An unexpected error occurred during training for '{label}': {e}")
            continue

    print(f"\nTraining process finished. PKL models saved.")
    print(f"Binary model export summary: {export_success_count} successful, {export_failure_count} failed/skipped.")
    if export_failure_count > 0:
        print("Check warnings/errors above for details on failed binary exports.")
    return models

if __name__ == '__main__':
    print("Starting GMM-HMM training and binary export process...")
    # 确保模型目录存在
    if not os.path.exists(MODEL_DIR):
        print(f"Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)

    # 检查数据目录是否存在
    print(f"Checking data directory: {DATA_DIR}")
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please create it and populate it with subdirectories containing WAV files for each keyword/background.")
         print("Example structure:")
         print(f"{DATA_DIR}/")
         print("  ├── keyword1/")
         print("  │   ├── file1.wav")
         print("  │   └── file2.wav")
         print("  └── background/")
         print("      └── noise1.wav")
    else:
        # 确保 utils.py 可用
        if 'load_data' not in globals():
             print("Error: Could not import 'load_data' from 'utils'. Please ensure 'utils.py' is in the same directory or accessible.")
        else:
            print(f"Training models using data from '{DATA_DIR}' and saving to '{MODEL_DIR}'...")
            train_models(DATA_DIR, MODEL_DIR, FEATURE_PARAMS, HMM_PARAMS)