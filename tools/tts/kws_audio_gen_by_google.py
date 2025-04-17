"""
使用 Google Cloud Text-to-Speech API 生成唤醒词的多种音色音频，支持多线程并行处理
"""
import os
import argparse
import requests
import json
import base64
import random
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import subprocess
import tempfile
from google.cloud import texttospeech
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# 默认可用的声音列表（语言、音色、性别组合）
DEFAULT_VOICES = [
    # 中文普通话女声
    {"name": "cmn-CN-Standard-A", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    {"name": "cmn-CN-Standard-B", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Standard-C", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Standard-D", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    # 中文普通话神经网络
    {"name": "cmn-CN-Wavenet-A", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    {"name": "cmn-CN-Wavenet-B", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Wavenet-C", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Wavenet-D", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    # 台湾中文
    {"name": "cmn-TW-Standard-A", "language_code": "cmn-TW", "ssml_gender": "FEMALE"},
    {"name": "cmn-TW-Standard-B", "language_code": "cmn-TW", "ssml_gender": "MALE"},
    {"name": "cmn-TW-Standard-C", "language_code": "cmn-TW", "ssml_gender": "MALE"},
    # 台湾中文神经网络
    {"name": "cmn-TW-Wavenet-A", "language_code": "cmn-TW", "ssml_gender": "FEMALE"},
    {"name": "cmn-TW-Wavenet-B", "language_code": "cmn-TW", "ssml_gender": "MALE"},
    {"name": "cmn-TW-Wavenet-C", "language_code": "cmn-TW", "ssml_gender": "MALE"},
    # 中文普通话Studio (更高质量)
    {"name": "cmn-CN-Studio-A", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    {"name": "cmn-CN-Studio-B", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Studio-C", "language_code": "cmn-CN", "ssml_gender": "MALE"},
    {"name": "cmn-CN-Studio-D", "language_code": "cmn-CN", "ssml_gender": "FEMALE"},
    # 粤语
    {"name": "yue-HK-Standard-A", "language_code": "yue-HK", "ssml_gender": "FEMALE"},
    {"name": "yue-HK-Standard-B", "language_code": "yue-HK", "ssml_gender": "MALE"},
    {"name": "yue-HK-Standard-C", "language_code": "yue-HK", "ssml_gender": "FEMALE"},
    {"name": "yue-HK-Standard-D", "language_code": "yue-HK", "ssml_gender": "MALE"},
    # 英语（美国，用于测试中英混合的场景）
    {"name": "en-US-Standard-A", "language_code": "en-US", "ssml_gender": "MALE"},
    {"name": "en-US-Standard-B", "language_code": "en-US", "ssml_gender": "MALE"},
    {"name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": "FEMALE"},
    {"name": "en-US-Standard-D", "language_code": "en-US", "ssml_gender": "MALE"},
    {"name": "en-US-Standard-E", "language_code": "en-US", "ssml_gender": "FEMALE"},
    {"name": "en-US-Standard-F", "language_code": "en-US", "ssml_gender": "FEMALE"},
    # 日语（用于测试多语言场景）
    {"name": "ja-JP-Standard-A", "language_code": "ja-JP", "ssml_gender": "FEMALE"},
    {"name": "ja-JP-Standard-B", "language_code": "ja-JP", "ssml_gender": "MALE"},
    {"name": "ja-JP-Standard-C", "language_code": "ja-JP", "ssml_gender": "MALE"},
    {"name": "ja-JP-Standard-D", "language_code": "ja-JP", "ssml_gender": "MALE"},
]

# 相似发音词列表 - 用于提高唤醒词的识别准确率
SIMILAR_SOUND_WORDS = {
    "小智同学": ["小志同学", "小纸同学", "小知同修", "晓之同学", "小织同学", "小治同学", "校址同学", "消脂同学", "肖智同学", "小值同学"]
}

# 音频设置选项
AUDIO_CONFIGS = [
    {"encoding": "mp3", "sample_rate_hertz": 16000, "pitch": 0.0, "speaking_rate": 1.0},
    {"encoding": "mp3", "sample_rate_hertz": 16000, "pitch": -2.0, "speaking_rate": 0.9},
    {"encoding": "mp3", "sample_rate_hertz": 16000, "pitch": 2.0, "speaking_rate": 1.1},
    {"encoding": "mp3", "sample_rate_hertz": 16000, "pitch": 0.0, "speaking_rate": 0.8},
    {"encoding": "mp3", "sample_rate_hertz": 16000, "pitch": 0.0, "speaking_rate": 1.2},
]

# 全局计数器和锁
counter_lock = Lock()
successful_count = 0
total_count = 0
print_lock = Lock()

# 凭证缓存
credentials_cache = None
credentials_lock = Lock()
client_cache = None
client_lock = Lock()

def get_credentials():
    """获取并缓存凭证，避免重复读取文件"""
    global credentials_cache
    with credentials_lock:
        if credentials_cache is None:
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            credentials_cache = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        return credentials_cache

def get_tts_client():
    """获取并缓存TTS客户端，避免重复创建"""
    global client_cache
    with client_lock:
        if client_cache is None:
            client_cache = texttospeech.TextToSpeechClient()
        return client_cache

def get_available_voices(filter_languages=None):
    """
    动态获取Google TTS支持的声音列表
    
    Args:
        filter_languages: 语言代码列表，如果提供则只返回这些语言的声音
        
    Returns:
        voices_list: 声音配置列表，格式与DEFAULT_VOICES相同
    """
    try:
        print("正在从Google TTS API获取可用声音列表...")
        
        # 获取TTS客户端
        client = get_tts_client()
        
        # 调用API获取支持的声音
        response = client.list_voices()
        
        # 解析响应
        voices_list = []
        for voice in response.voices:
            # 提取语言代码
            language_code = voice.language_codes[0]
            
            # 如果指定了过滤语言，跳过不匹配的
            if filter_languages and language_code not in filter_languages:
                continue
            
            # 获取性别
            ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
            
            # 添加到列表
            voices_list.append({
                "name": voice.name,
                "language_code": language_code,
                "ssml_gender": ssml_gender
            })
        
        print(f"成功获取 {len(voices_list)} 个可用声音")
        return voices_list
    
    except Exception as e:
        print(f"获取可用声音失败: {e}")
        print("将使用默认声音列表")
        return DEFAULT_VOICES

def generate_tts_task(task):
    """线程任务函数，处理单个TTS生成任务"""
    global successful_count, total_count
    
    text = task["text"]
    voice_config = task["voice_config"]
    audio_config = task["audio_config"] 
    output_file = task["output_file"]
    session = task["session"]
    voice_name = voice_config["name"]
    index = task["index"]
    count = task["count"]
    
    with print_lock:
        print(f"正在生成 {voice_name} ({index}/{count})...")
    
    success = generate_tts(text, voice_config, audio_config, output_file, session)
    
    with counter_lock:
        total_count += 1
        if success:
            successful_count += 1
            with print_lock:
                print(f"成功生成: {os.path.basename(output_file)}")
    
    return success

def generate_tts(text, voice_config, audio_config, output_file, session):
    """生成单个TTS音频文件, 先获取mp3再转为WAV"""
    try:
        # 获取凭证（使用缓存）
        credentials = get_credentials()
        
        # 创建一个当前线程专用的会话
        thread_session = requests.Session()
        thread_session.proxies.update(session.proxies)
        
        # 刷新token
        auth_req = Request(session=thread_session)
        credentials.refresh(auth_req)
        
        # 准备请求 (请求mp3格式)
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        voice_params = {
            "languageCode": voice_config["language_code"],
            "name": voice_config["name"],
            "ssmlGender": voice_config["ssml_gender"]
        }
        
        # 注意: audio_config 现在指定 mp3, sample_rate_hertz 可能会被忽略
        audio_params = {
            "audioEncoding": audio_config["encoding"],
            "sampleRateHertz": audio_config["sample_rate_hertz"], # 虽指定16k, 但mp3可能覆盖
            "pitch": audio_config["pitch"],
            "speakingRate": audio_config["speaking_rate"]
        }
        
        data = {
            "input": {"text": text},
            "voice": voice_params,
            "audioConfig": audio_params
        }
        
        # 发送请求
        response = thread_session.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            # 处理响应 - 获取mp3内容
            mp3_audio_content = base64.b64decode(response.json()["audioContent"])
            
            # 创建临时mp3文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3_file:
                tmp_mp3_path = tmp_mp3_file.name
                tmp_mp3_file.write(mp3_audio_content)
            
            try:
                # 使用ffmpeg进行转换: mp3 -> wav (16k, 16bit, mono)
                # output_file 的后缀应为 .wav
                ffmpeg_command = [
                    'ffmpeg',
                    '-i', tmp_mp3_path,       # 输入临时mp3文件
                    '-ar', '16000',           # 设置采样率为16kHz
                    '-ac', '1',               # 设置声道为单声道
                    '-acodec', 'pcm_s16le',   # 设置编码为16位PCM little-endian
                    '-y',                     # 覆盖输出文件（如果已存在）
                    output_file               # 输出WAV文件路径
                ]
                # print(f"Executing: {' '.join(ffmpeg_command)}") # 调试时可以取消注释
                process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                # print(f"FFmpeg stdout: {process.stdout}") # 调试时可以取消注释
                # print(f"FFmpeg stderr: {process.stderr}") # 调试时可以取消注释
                os.remove(tmp_mp3_path) # 转换成功后删除临时文件
                return True
            except subprocess.CalledProcessError as e:
                 with print_lock:
                    print(f"FFmpeg 转换失败 for {os.path.basename(output_file)}: {e}")
                    print(f"FFmpeg stderr: {e.stderr}")
                 os.remove(tmp_mp3_path) # 即使失败也尝试删除临时文件
                 return False
            except FileNotFoundError:
                 with print_lock:
                     print(f"错误：找不到 ffmpeg 命令。请确保已安装 ffmpeg 并将其添加至系统 PATH。")
                 os.remove(tmp_mp3_path) # 尝试删除临时文件
                 return False
            except Exception as e:
                with print_lock:
                    print(f"转换或删除临时文件时出错 for {os.path.basename(output_file)}: {e}")
                # 尝试删除临时文件，以防万一
                if os.path.exists(tmp_mp3_path):
                    os.remove(tmp_mp3_path)
                return False
        else:
            with print_lock:
                print(f"Google TTS 请求失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        with print_lock:
            print(f"生成或处理失败 for {os.path.basename(output_file)}: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成唤醒词的多种音色音频')
    parser.add_argument('--use-env-proxy', action='store_true', default=True, help='使用环境变量中的代理')
    parser.add_argument('--use-socks5-proxy', action='store_true', help='使用SOCKS5代理')
    parser.add_argument('--socks5-proxy', default='socks5://127.0.0.1:1080', help='SOCKS5代理地址')
    parser.add_argument('--text', default="小智同学", help='要转换为语音的文本')
    parser.add_argument('--output-dir', default="./data/kws/", help='输出目录')
    parser.add_argument('--count', type=int, default=5, help='每种音色生成的样本数量')
    parser.add_argument('--threads', type=int, default=16, help='并行线程数')
    parser.add_argument('--dynamic-voices', action='store_true', help='动态获取支持的声音列表')
    parser.add_argument('--filter-languages', nargs='+', help='仅生成指定语言的声音，例如：cmn-CN yue-HK')
    parser.add_argument('--generate-similar', action='store_true', help='同时生成相似发音词的音频，提高识别准确率')
    parser.add_argument('--similar-words', nargs='+', help='自定义相似发音词列表，不提供则使用默认列表')
    args = parser.parse_args()

    # 确保线程数不超过合理范围
    max_threads = min(args.threads, os.cpu_count() * 2)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取代理设置
    proxies = {}
    
    if args.use_env_proxy:
        http_proxy = os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('HTTPS_PROXY')
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        if http_proxy or https_proxy:
            print(f"使用环境变量代理: HTTP_PROXY={http_proxy}, HTTPS_PROXY={https_proxy}")
    
    if args.use_socks5_proxy:
        proxies = {
            'http': args.socks5_proxy,
            'https': args.socks5_proxy
        }
        print(f"使用SOCKS5代理: {args.socks5_proxy}")
    
    # 创建主session并设置代理
    session = requests.Session()
    if proxies:
        session.proxies.update(proxies)
    
    # 获取声音列表
    if args.dynamic_voices:
        VOICES = get_available_voices(args.filter_languages)
    else:
        VOICES = DEFAULT_VOICES
        if args.filter_languages:
            VOICES = [v for v in VOICES if v["language_code"] in args.filter_languages]
            print(f"已过滤声音列表，保留 {len(VOICES)} 个声音")
    
    # 如果没有可用声音，提前退出
    if not VOICES:
        print("错误：没有可用的声音!")
        return
    
    # 准备所有生成任务
    tasks = []
    
    # 处理目标唤醒词
    texts_to_generate = [args.text]
    
    # 如果需要生成相似发音词
    if args.generate_similar:
        if args.similar_words:
            texts_to_generate.extend(args.similar_words)
        elif args.text in SIMILAR_SOUND_WORDS:
            texts_to_generate.extend(SIMILAR_SOUND_WORDS[args.text])
            print(f"使用默认相似发音词列表: {SIMILAR_SOUND_WORDS[args.text]}")
        else:
            print(f"警告: 未找到'{args.text}'的相似发音词列表")
    
    # 为每个文本生成任务
    for text in texts_to_generate:
        for voice in VOICES:
            for i in range(args.count):
                # 随机选择一个音频配置
                audio_config = random.choice(AUDIO_CONFIGS)
                
                # 构建文件名
                voice_name = voice["name"]
                gender = voice["ssml_gender"].lower()
                pitch = int(audio_config["pitch"] * 10)  # 变为整数以用于文件名
                rate = int(audio_config["speaking_rate"] * 10)
                
                # 确定文件名前缀（原文本或相似发音词）
                prefix = "orig" if text == args.text else "similar"
                
                filename = f"{prefix}_{text}_{voice_name}_{gender}_p{pitch}_r{rate}_{i+1}.wav"
                output_path = os.path.join(args.output_dir, filename)
                
                # 添加到任务列表
                tasks.append({
                    "text": text,
                    "voice_config": voice,
                    "audio_config": audio_config,
                    "output_file": output_path,
                    "session": session,
                    "index": i+1,
                    "count": args.count
                })
    
    # 显示配置信息
    print(f"准备生成 {len(tasks)} 个音频文件，使用 {max_threads} 个并行线程")
    start_time = time.time()
    
    # 使用线程池并行处理任务
    global successful_count, total_count
    successful_count = 0
    total_count = 0
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_tts_task, task) for task in tasks]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    # 计算执行时间
    elapsed_time = time.time() - start_time
    
    print(f"生成完成! 共生成 {successful_count}/{total_count} 个音频文件")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"文件保存在: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()