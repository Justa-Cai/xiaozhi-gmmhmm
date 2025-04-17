"""
使用 Google Cloud Text-to-Speech API 生成唤醒词的负样本音频数据，支持多线程并行处理
"""
import os
import argparse
import requests
import json
import base64
import random
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import subprocess
import shutil
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
]

# 音频设置选项
AUDIO_CONFIGS = [
    {"encoding": "MP3", "sample_rate_hertz": 16000, "pitch": 0.0, "speaking_rate": 1.0},
    {"encoding": "MP3", "sample_rate_hertz": 16000, "pitch": -2.0, "speaking_rate": 0.9},
    {"encoding": "MP3", "sample_rate_hertz": 16000, "pitch": 2.0, "speaking_rate": 1.1},
    {"encoding": "MP3", "sample_rate_hertz": 24000, "pitch": 0.0, "speaking_rate": 0.8},
    {"encoding": "MP3", "sample_rate_hertz": 24000, "pitch": 0.0, "speaking_rate": 1.2},
]

# 负样本文本模板
NEGATIVE_TEXT_TEMPLATES = {
    "cmn-CN": [
        "今天天气真不错",
        "我想去公园散步",
        "明天有什么计划",
        "这个菜很好吃",
        "我喜欢听音乐",
        "最近有什么新闻",
        "你觉得这个怎么样",
        "帮我查一下时间",
        "这个电影很精彩",
        "请打开电视机",
        "我需要买一些水果",
        "这本书写得很好",
        "附近有什么好餐厅",
        "昨天发生了什么事",
        "周末我想去旅行",
        "能告诉我路怎么走吗",
        "这个问题很难解决",
        "我想学习一门新技能",
        "这个地方风景优美",
        "现在几点钟了",
        "可以帮我预订机票吗",
        "这个商品多少钱",
        "今年的目标是什么",
        "最近工作很忙",
        "我喜欢这种颜色",
        "这个设计很独特",
        "音乐声音有点大",
        "需要添加一些调料",
        "明天会下雨吗",
        "这个季节很舒适"
    ],
    "cmn-TW": [
        "今天天氣真不錯",
        "我想去公園散步",
        "明天有什麼計劃",
        "這個菜很好吃",
        "我喜歡聽音樂",
        "最近有什麼新聞",
        "你覺得這個怎麼樣",
        "幫我查一下時間",
        "這個電影很精彩",
        "請打開電視機",
        "我需要買一些水果",
        "這本書寫得很好",
        "附近有什麼好餐廳",
        "昨天發生了什麼事",
        "週末我想去旅行",
        "能告訴我路怎麼走嗎",
        "這個問題很難解決",
        "我想學習一門新技能",
        "這個地方風景優美",
        "現在幾點鐘了",
        "可以幫我預訂機票嗎",
        "這個商品多少錢",
        "今年的目標是什麼",
        "最近工作很忙",
        "我喜歡這種顏色",
        "這個設計很獨特",
        "音樂聲音有點大",
        "需要添加一些調料",
        "明天會下雨嗎",
        "這個季節很舒適"
    ],
}

# 移除相似发音词列表，改为日常常用词列表
DAILY_WORDS = [
    "今天", "明天", "昨天", "早上", "中午", "晚上",
    "吃饭", "睡觉", "工作", "学习", "旅行", "购物",
    "电影", "音乐", "书籍", "新闻", "体育", "游戏",
    "朋友", "家人", "同事", "老师", "学生", "医生",
    "手机", "电脑", "电视", "相机", "汽车", "自行车",
    "天气", "季节", "春天", "夏天", "秋天", "冬天",
    "北京", "上海", "广州", "深圳", "杭州", "成都",
    "咖啡", "茶", "水果", "蔬菜", "海鲜", "肉类",
    "公司", "学校", "医院", "银行", "超市", "餐厅",
    "问题", "答案", "方案", "建议", "意见", "决定"
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

def check_ffmpeg():
    """检查 ffmpeg 是否安装并可用"""
    if shutil.which("ffmpeg") is None:
        print("错误：找不到 ffmpeg 命令。请确保已安装 ffmpeg 并将其添加到系统 PATH。")
        print("安装说明:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("  macOS (Homebrew): brew install ffmpeg")
        print("  Windows: 从 https://ffmpeg.org/download.html 下载并添加到 PATH")
        exit(1)
    print("检测到 ffmpeg 命令。")

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

def get_negative_samples(target_word, mode="daily", similar_words=None, count=10):
    """
    生成负样本文本列表
    
    Args:
        target_word: 目标唤醒词
        mode: 负样本生成模式，"templates"使用模板句子，"daily"使用日常词语，"both"两者都用
        similar_words: 相似发音词列表(不再使用)
        count: 需要生成的样本数量
        
    Returns:
        samples: 负样本文本列表
    """
    samples = []
    
    # 根据模式生成样本
    if mode in ["templates", "both"]:
        for lang_code in NEGATIVE_TEXT_TEMPLATES:
            templates = NEGATIVE_TEXT_TEMPLATES[lang_code]
            for template in templates:
                samples.append({"text": template, "type": "template", "language": lang_code})
    
    if mode in ["daily", "both"]:
        for word in DAILY_WORDS:
            samples.append({"text": word, "type": "daily", "language": "cmn-CN"})
    
    # 如果样本不足，重复使用现有样本
    if len(samples) < count:
        samples = samples * (count // len(samples) + 1)
    
    # 随机打乱并截取需要的数量
    random.shuffle(samples)
    return samples[:count]

def generate_tts_task(task):
    """线程任务函数，处理单个TTS生成和转换任务"""
    global successful_count, total_count
    
    text = task["text"]
    voice_config = task["voice_config"]
    audio_config = task["audio_config"] 
    temp_mp3_file = task["temp_mp3_file"]
    output_wav_file = task["output_wav_file"]
    session = task["session"]
    voice_name = voice_config["name"]
    index = task["index"]
    count = task["count"]
    sample_type = task.get("sample_type", "unknown")
    
    with print_lock:
        print(f"正在生成负样本 ({sample_type}) {voice_name} ({index}/{count}) -> {os.path.basename(output_wav_file)}...")
    
    success = generate_tts_and_convert(text, voice_config, audio_config, temp_mp3_file, output_wav_file, session)
    
    with counter_lock:
        total_count += 1
        if success:
            successful_count += 1
            with print_lock:
                print(f"成功生成并转换: {os.path.basename(output_wav_file)}")
    
    return success

def generate_tts_and_convert(text, voice_config, audio_config, temp_mp3_file, output_wav_file, session):
    """生成单个TTS音频文件，并转换为 WAV 格式"""
    try:
        # --- 1. 下载 MP3 ---
        # 获取凭证（使用缓存）
        credentials = get_credentials()
        
        # 创建一个当前线程专用的会话
        thread_session = requests.Session()
        thread_session.proxies.update(session.proxies)
        
        # 刷新token
        auth_req = Request(session=thread_session)
        credentials.refresh(auth_req)
        
        # 准备请求
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
        
        # 注意：仍然请求 MP3 格式，因为 API 可能不支持直接输出 WAV
        # 或许可以研究下API是否支持LINEAR16，但目前先保持 MP3 -> WAV 转换流程
        audio_params = {
            "audioEncoding": "MP3", # 保持 MP3 请求
            "sampleRateHertz": audio_config["sample_rate_hertz"], # 这个采样率会被 ffmpeg 覆盖
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
            # 处理响应 - 保存为临时 MP3
            audio_content = base64.b64decode(response.json()["audioContent"])
            os.makedirs(os.path.dirname(temp_mp3_file), exist_ok=True) # 确保临时目录存在
            with open(temp_mp3_file, "wb") as out:
                out.write(audio_content)
        else:
            with print_lock:
                print(f"下载 MP3 失败: {response.status_code} - {response.text}")
            return False

        # --- 2. 使用 ffmpeg 转换为 WAV ---
        try:
            # ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav
            command = [
                "ffmpeg",
                "-i", temp_mp3_file,
                "-acodec", "pcm_s16le", # 16位 PCM
                "-ac", "1",            # 单声道
                "-ar", "16000",         # 16kHz 采样率
                output_wav_file
            ]
            # 使用 subprocess.run，设置 capture_output=True 来捕获输出和错误
            result = subprocess.run(command, check=True, capture_output=True, text=True) 
            
            # 转换成功后删除临时 MP3 文件
            os.remove(temp_mp3_file)
            return True
            
        except subprocess.CalledProcessError as e:
            with print_lock:
                print(f"ffmpeg 转换失败: {output_wav_file}")
                print(f"  命令: {' '.join(e.cmd)}")
                print(f"  返回码: {e.returncode}")
                print(f"  错误输出: {e.stderr}")
            # 保留失败的 MP3 文件用于调试
            # if os.path.exists(temp_mp3_file):
            #     print(f"  保留临时 MP3 文件: {temp_mp3_file}")
            return False
        except FileNotFoundError:
             with print_lock:
                print(f"错误：ffmpeg 命令未找到。请确保已安装并添加到 PATH。")
             # 尝试删除可能存在的损坏的 MP3
             if os.path.exists(temp_mp3_file):
                 try:
                     os.remove(temp_mp3_file)
                 except OSError:
                     pass
             return False


    except Exception as e:
        with print_lock:
            print(f"生成或转换过程中发生错误: {e}")
        # 尝试清理临时文件
        if 'temp_mp3_file' in locals() and os.path.exists(temp_mp3_file):
            try:
                os.remove(temp_mp3_file)
            except OSError:
                pass
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成唤醒词的负样本音频数据 (WAV 格式)')
    parser.add_argument('--use-env-proxy', action='store_true', default=True, help='使用环境变量中的代理')
    parser.add_argument('--use-socks5-proxy', action='store_true', help='使用SOCKS5代理')
    parser.add_argument('--socks5-proxy', default='socks5://127.0.0.1:1080', help='SOCKS5代理地址')
    parser.add_argument('--target', default="小智同学", help='目标唤醒词')
    parser.add_argument('--output-dir', default="./data/negative_samples", help='输出目录')
    parser.add_argument('--count', type=int, default=100, help='生成的负样本总数量')
    parser.add_argument('--threads', type=int, default=16, help='并行线程数')
    parser.add_argument('--mode', default="both", choices=["templates", "daily", "both"], 
                       help='负样本生成模式：templates(使用模板句子)，daily(使用日常词语)，both(两者都用)')
    parser.add_argument('--filter-languages', nargs='+', default=["cmn-CN", "cmn-TW"], 
                       help='仅使用指定语言的声音，例如：cmn-CN cmn-TW')
    args = parser.parse_args()

    # 检查 ffmpeg 是否可用
    check_ffmpeg()

    # 确保线程数不超过合理范围
    max_threads = min(args.threads, os.cpu_count() * 2)
    
    # 创建输出目录和临时目录
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_mp3")
    os.makedirs(temp_dir, exist_ok=True)
    
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
    
    # 过滤声音列表
    VOICES = [v for v in DEFAULT_VOICES if v["language_code"] in args.filter_languages]
    print(f"已过滤声音列表，使用 {len(VOICES)} 个声音")
    
    # 如果没有可用声音，提前退出
    if not VOICES:
        print("错误：没有可用的声音!")
        return
    
    # 生成负样本文本
    negative_samples = get_negative_samples(
        args.target, 
        mode=args.mode, 
        count=args.count
    )
    
    print(f"已生成 {len(negative_samples)} 个负样本文本")
    
    # 准备所有生成任务
    tasks = []
    sample_index = 0
    
    for sample in negative_samples:
        sample_index += 1
        # 为每个样本随机选择一个声音和音频配置
        voice_config = random.choice(VOICES)
        # 确保语言匹配
        if sample["language"].startswith(voice_config["language_code"].split("-")[0]):
            audio_config = random.choice(AUDIO_CONFIGS)
            
            # 构建文件名
            voice_name = voice_config["name"]
            gender = voice_config["ssml_gender"].lower()
            sample_type = sample["type"]
            text = sample["text"]
            
            # 创建文件名，避免过长和特殊字符，后缀改为 .wav
            safe_text = text.replace(" ", "_").replace("，", "").replace("。", "")[:20]
            base_filename = f"neg_{sample_type}_{safe_text}_{voice_name}_{gender}_{sample_index}"
            output_wav_filename = f"{base_filename}.wav"
            temp_mp3_filename = f"{base_filename}.mp3"
            
            output_wav_path = os.path.join(args.output_dir, output_wav_filename)
            temp_mp3_path = os.path.join(temp_dir, temp_mp3_filename)
            
            # 添加到任务列表
            tasks.append({
                "text": text,
                "voice_config": voice_config,
                "audio_config": audio_config,
                "temp_mp3_file": temp_mp3_path,
                "output_wav_file": output_wav_path,
                "session": session,
                "index": sample_index,
                "count": len(negative_samples),
                "sample_type": sample_type
            })
    
    # 显示配置信息
    print(f"准备生成并转换 {len(tasks)} 个负样本音频文件 (WAV 格式)，使用 {max_threads} 个并行线程")
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
    
    print(f"生成并转换完成! 共生成 {successful_count}/{total_count} 个 WAV 负样本音频文件")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"文件保存在: {os.path.abspath(args.output_dir)}")
    
    # 清理空的临时目录（如果存在且为空）
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
        elif os.path.exists(temp_dir) and os.listdir(temp_dir):
             print(f"警告：临时目录 {temp_dir} 中仍有文件，可能部分转换失败。请手动检查。")

    except OSError as e:
         print(f"清理临时目录时出错: {e}")

if __name__ == "__main__":
    main() 