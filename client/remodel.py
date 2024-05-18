import os
import time

path = r"G:\models\coquiai\GPT_XTTS_LJSpeech_FT-May-16-2024_06+59PM-a6e2beb1"


def check_files(directory):
    # 获取当前时间
    current_time = time.time()

    # 定义时间阈值，5分钟前的时间（单位：秒）
    time_threshold = current_time - 5 * 60

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 使用正则表达式匹配文件名
        import re
        # best_model_8646.pth
        if re.match(r'best_model_([0-9]+)\.pth', filename):
            # 获取文件的完整路径
            file_path = os.path.join(directory, filename)

            # 获取文件的创建时间（注意：os.path.getctime返回的是文件的创建或上次状态更改的时间，单位为浮点数的秒）
            creation_time = os.path.getctime(file_path)
            new_name = os.path.basename(file_path)
            new_name = new_name.replace("_", "-")
            new_name = new_name.replace(".pth", ".pth.back")

            # 检查文件创建时间是否在5分钟之前
            if creation_time < time_threshold:
                os.rename(file_path, os.path.join(directory, new_name))
                print(f" {filename} -> {new_name}")


while True:
    check_files(path)
    time.sleep(60)
