# 唤醒词音频生成工具

这个工具用于生成"你好小智"唤醒词的音频数据，使用Google Cloud Text-to-Speech API来生成不同音色的MP3格式音频文件。

## 前提条件

1. 已安装Python 3.6+
2. 已安装所需依赖项：
   ```
   pip install -r requirements.txt
   ```
3. 已配置Google Cloud认证：
   - 安装Google Cloud SDK
   - 运行 `gcloud auth application-default login` 进行认证
   - 或下载服务账号凭证JSON文件，然后设置环境变量：
     ```
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-project-credentials.json"
     ```
4. 确保有一个可用的SOCKS5代理服务（默认地址：127.0.0.1:1080）

## 使用方法

### 基本用法

```bash
python generate_kws_audio.py
```

这将生成默认唤醒词"你好小智"的多个音频样本，保存在`./kws_audio`目录下。

### 高级选项

```bash
python generate_kws_audio.py --text "你好小智" --output "./my_audio_dir" --count 10 --language "cmn-CN" --no-proxy
```

参数说明：
- `--text`：要生成的文本内容，默认为"你好小智"
- `--output`：音频文件输出目录，默认为"./kws_audio"
- `--language`：语言代码，默认为"cmn-CN"（中文普通话）
- `--count`：每种音色生成的样本数量，默认为5
- `--no-proxy`：不使用代理直接连接Google服务

## 输出文件

生成的音频文件将以以下格式命名：
```
kws_<声音名称>_<随机ID>.mp3
```

例如：`kws_cmn-CN-Wavenet-D_a1b2c3d4.mp3`

## 注意事项

1. 生成的音频会有不同的音色、语速、音调和音量，以增加数据的多样性
2. 如果没有SOCKS5代理，可以使用`--no-proxy`参数
3. 请确保Google Cloud项目已启用Text-to-Speech API 