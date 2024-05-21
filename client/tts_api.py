import base64
import os
import re
import time
import uuid
from io import BytesIO

import torch
import numpy
import soundfile as sf
import stable_whisper

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fast_config import init_config, Config, ServerConfig, ModleConfig

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    description='AI TTS 接口文档',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redocs'
)


class ModelCache:
    config: XttsConfig = None
    model: Xtts = None
    svr_config: ModleConfig = None
    gpt_cond_latent: torch.Tensor = None
    gpt_cond_latent_short: torch.Tensor = None
    speaker_embedding: torch.Tensor = None
    speaker_embedding_short: torch.Tensor = None


server_config: Config = None
models_cache: dict[str, ModelCache] = {}


class RequestTTS(BaseModel):
    text: str = ""
    lang: str = ""
    out_base64: bool = False


splits = {",", ".", "?", "!", "~", ":", "—", "..."}
reps = {
    '，': ',',
    '。': '.',
    '？': '?',
    '！': '!',
    '；': ';',
    '：': ':',
    '…': '...',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '《': '<',
    '》': '>',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '、': ',',
    '—': '-',
    '～': '~',
    '·': '.',
    '「': '"',
    '」': '"',
    '『': "'",
    '』': "'",
    '〈': '<',
    '〉': '>',
    '﹁': '"',
    '﹂': '"',
    '﹃': "'",
    '﹄': "'",
}


def cut_text(text: str, restrict: int = 250):
    # 首先将常用符号换为英文
    for k, v in reps.items():
        text = text.replace(k, v)
    # 然后将文本按照标点符号切分
    re_list = []
    inp = text.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
        if len(mergeitems[-1]) > restrict:
            return None
    opt = "\n".join(mergeitems)
    return opt
    # if len(text) > restrict:
    #     # 长度大于restrict的文本进行切分
    #     for i in range(0, len(text), restrict):
    #         re_list.append(text[i:i + restrict])
    #         if len(re_list[-1]) > restrict:
    #             return None
    # else:
    #     re_list.append(text)
    #     if len(re_list[-1]) > restrict:
    #         return None
    # return re_list


@app.post("/v1/api/tts")
async def tts(item: RequestTTS):
    """
    TTS合成
    Args:
        item: RequestTTS

    Returns:

    """
    if item.lang not in models_cache.keys():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Language not supported")
    # 处理切分文本
    re_list = []
    text = item.text.strip()
    text = text.strip("\n")
    re_list = cut_text(text)
    if re_list is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text too long")
    re_list = re_list.replace("\n\n", "\n")
    re_list = re_list.split("\n")
    zero_wav = numpy.zeros(
        int(0.3 * 24000),
        dtype=numpy.float16
    )
    audio_opt = []
    for text_i in re_list:
        with torch.no_grad():
            # if (text[0] not in splits and len(
            #   get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
            txt = text_i.strip()
            if txt == "":
                continue
            print("开始推理:", txt)
            if len(txt) < 1:
                return JSONResponse(status_code=200, content={"data": ""})
            if len(txt) < 10:
                out = models_cache[item.lang].model.inference(
                    txt,
                    models_cache[item.lang].svr_config.language,
                    models_cache[item.lang].gpt_cond_latent_short,
                    models_cache[item.lang].speaker_embedding_short,
                )
            else:
                out = models_cache[item.lang].model.inference(
                    txt,
                    models_cache[item.lang].svr_config.language,
                    models_cache[item.lang].gpt_cond_latent,
                    models_cache[item.lang].speaker_embedding,
                )
            audio_opt.append(out["wav"])
            audio_opt.append(zero_wav)

    out = (numpy.concatenate(audio_opt, 0) * 32768).astype(
        numpy.int16
    )
    # out = numpy.concatenate(audio_opt, 0)
    in_memory_wav = BytesIO()
    # wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
    with sf.SoundFile(in_memory_wav, mode='w', samplerate=24000, channels=1, format='wav') as audio_file:
        audio_file.write(out)
    in_memory_wav.seek(0)
    if item.out_base64:
        return JSONResponse(status_code=200,
                            content={"data": base64.b64encode(in_memory_wav.getbuffer()).decode()}
                            )
    return StreamingResponse(in_memory_wav, media_type="audio/wav")


@app.post("/v1/api/up_file")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件,
    Args:
        file: 上传的文件

    Returns: 返回创建的文件名

    """
    if not os.path.exists(f"upload"):
        os.makedirs("upload")
    # 生成uuid
    u = str(uuid.uuid4()).replace("-", "")
    # 生成时间戳
    t = str(int(time.time()))
    # 保存文件
    with open(f"upload/{u}_{t}", "wb") as f:
        f.write(file.file.read())
    return {"name": u}


class RequestASR(BaseModel):
    """
    ASR请求模型类

    Attributes:
        file_name: 需要进行语音识别的文件名。
        out_type:
    """
    file_name: str = ""
    out_type: str = "obj"
    align_text: str = ""
    lang: str = "en"


import asr_help

stable_whisper_model = None


@app.post("/v1/api/asr")
async def asr(item: RequestASR):
    """
    处理语音识别的API请求。

    Args:
        item (RequestASR): 包含文件名和输出类型信息的ASR请求对象。

    Returns:
        无返回值，但会异步处理语音识别任务。

    Raises:
        HTTPException: 如果请求的文件不存在，则抛出400状态码的HTTP异常。
    """
    # 检查需要转写的文件是否存在
    inpf = "upload/{}".format(item.file_name)
    if not os.path.exists(inpf):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File not found")
    global stable_whisper_model
    if stable_whisper_model is None:
        stable_whisper_model = stable_whisper.load_faster_whisper(server_config.model_whisper.model_path, device="cuda",
                                                                  compute_type="int8_float16")
    with torch.no_grad():
        if item.align_text != "":
            result = stable_whisper_model.align(inpf, item.align_text, language=item.lang)
        else:
            result = stable_whisper_model.transcribe_stable(inpf, language=item.lang)
    if item.out_type == "obj":
        return JSONResponse(status_code=200,
                            content={
                                "data": asr_help.raw_json_to_non_json(result.to_dict())
                            }
                            )
    elif item.out_type == "vtt":
        return JSONResponse(status_code=200,
                            content={
                                "data": asr_help.to_vtt_data(inpf, result)
                            }
                            )


def initsvr():
    global server_config, model_config
    server_config = init_config("config.json")
    # model_config = XttsConfig()
    # model_config.load_json(config_path)
    for lang, model in server_config.models.items():
        print(f"Loading model {lang}")
        model_cache = ModelCache()
        model_cache.svr_config = model
        model_cache.config = XttsConfig()
        model_cache.config.load_json(model.config_path)
        model_cache.model = Xtts.init_from_config(model_cache.config)
        # model_cache.model.load_checkpoint(model_cache.config, checkpoint_path=model.checkpoint_path,
        #                                   vocab_path=model.tokenizer_path, use_deepspeed=False)
        model_cache.model.load_checkpoint(model_cache.config,
                                          checkpoint_path=model.checkpoint_path,
                                          vocab_path=model.tokenizer_path,
                                          use_deepspeed=False)
        model_cache.gpt_cond_latent, model_cache.speaker_embedding = model_cache.model.get_conditioning_latents(
            audio_path=[model.speaker_reference]
        )
        print("Model s")
        model_cache.gpt_cond_latent_short, model_cache.speaker_embedding_short = model_cache.model.get_conditioning_latents(
            audio_path=[model.speaker_reference_short]
        )
        model_cache.model = model_cache.model.cuda()
        models_cache[lang] = model_cache

import uvicorn

if __name__ == '__main__':
    initsvr()
    print("loading done")
    uvicorn.run(app, host=server_config.server.host, port=server_config.server.port)
