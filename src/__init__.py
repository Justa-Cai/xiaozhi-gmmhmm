# src/__init__.py

# 从子模块导入，使其在 src 包级别可用
from .feature_extraction import extract_mfcc
from .train_gmm_hmm import train_models
from .realtime_inference import real_time_inference
from .utils import load_data

print("KWS src package initialized") # 可以加一句初始化信息（可选）

# 可以定义 __all__ 来控制 'from src import *' 的行为 (可选)
__all__ = ['extract_mfcc', 'train_models', 'real_time_inference', 'load_data'] 