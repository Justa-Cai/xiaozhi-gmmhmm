# GMM-HMM 关键词识别项目

本项目使用高斯混合模型-隐马尔可夫模型 (GMM-HMM) 实现简单的关键词识别（唤醒词检测）。

## 项目结构

```
gmm_hmm_keyword_spotting/
├── data/                 # 音频数据
│   ├── keyword1/         # 关键词1音频
│   ├── keyword2/         # 关键词2音频
│   └── background/       # 背景噪音
├── models/               # 训练好的模型
├── src/                  # 源代码
│   ├── __init__.py
│   ├── feature_extraction.py  # 特征提取
│   ├── train_gmm_hmm.py     # 训练脚本
│   ├── realtime_inference.py # 实时推理
│   └── utils.py             # 辅助函数
├── requirements.txt      # 依赖库
└── README.md             # 本文件
```

## 使用方法

1.  **准备数据**: 
    *   在 `data/` 目录下创建对应关键词和背景噪音的子目录。
    *   将相应的 `.wav` 音频文件放入各子目录中（确保是单声道, 16kHz采样率）。
2.  **安装依赖**: `pip install -r requirements.txt`
3.  **训练模型**: `python src/train_gmm_hmm.py`
4.  **实时测试**: `python src/realtime_inference.py`

## 注意

*   音频文件推荐使用 16kHz 采样率、16-bit 位深、单声道的 WAV 格式。
*   训练效果很大程度上取决于训练数据的质量和数量。
*   实时推理的参数（如能量阈值、检测窗口）可能需要根据实际环境进行调整。 