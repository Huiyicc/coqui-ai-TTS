# 将指定目录下的音频重新采样为指定采样率
import os
from pydub import AudioSegment

# 目标采样率
target_sr = 22050
# 源音频目录
source_dir = r'G:\dataset\VCTK-Corpus-0.92\wav48_silence_trimmed\p229'
# 输出音频目录
output_dir = r'G:\dataset\VCTK-Corpus-0.92\wav22_silence_trimmed\p229\wavs'

format_in = "flac"
format_out = "wav"

def resample_audio(input_file, output_file, sample_rate=24000):
    """
    将音频文件重新采样至指定的采样率，默认为24000Hz。

    :param input_file: 输入音频文件路径
    :param output_file: 输出音频文件路径
    :param sample_rate: 目标采样率，默认24000Hz
    """
    audio = AudioSegment.from_file(input_file)
    audio_resampled = audio.set_frame_rate(sample_rate)
    audio_resampled.export(output_file, format="wav")


# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 遍历源音频目录
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(f'.{format_in}'):
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_dir, file.split('.')[0] + f'.{format_out}')
            resample_audio(input_file, output_file, target_sr)
            print(f"Resampled {input_file} to {output_file}")
print("Resampling finished.")
