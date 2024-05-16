import base64
from io import BytesIO

import torch
import soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fast_config import init_config, Config, ServerConfig, ModleConfig

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()


class ModelCache:
    config: XttsConfig = None
    model: Xtts = None
    svr_config: ModleConfig = None
    gpt_cond_latent: torch.Tensor = None
    speaker_embedding: torch.Tensor = None


server_config: Config = None
models_cache: dict[str, ModelCache] = {}


class Item(BaseModel):
    text: str = ""
    lang: str = ""
    out_base64: bool = False

nn = 1.5
@app.post("/v1/api/tts")
async def tts(item: Item):
    global nn
    if item.lang not in models_cache.keys():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Language not supported")
    with torch.no_grad():
        txt = item.text.strip()
        if len(txt) < 1:
            return JSONResponse(status_code=200, content={"data": ""})
        if txt[0] == "/":
            nn = float(txt[1:])
        txt = "." + txt
        n = 1
        while len(txt) < 10:
            txt = txt + "... " + txt + "... "
            n += nn

        out = models_cache[item.lang].model.inference(
            txt,
            models_cache[item.lang].svr_config.language,
            models_cache[item.lang].gpt_cond_latent,
            models_cache[item.lang].speaker_embedding,
            speed=0.95,
        )
        out["wav"] = out["wav"][:int(len(out["wav"]) // n)]
        in_memory_wav = BytesIO()
        # wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        with sf.SoundFile(in_memory_wav, mode='w', samplerate=24000, channels=1, format='ogg') as audio_file:
            audio_file.write(out["wav"])
        in_memory_wav.seek(0)
        if item.out_base64:
            return JSONResponse(status_code=200,
                                content={"data": base64.b64encode(in_memory_wav.getbuffer()).decode()}
                                )
        return StreamingResponse(in_memory_wav, media_type="audio/wav")


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
        model_cache.model.load_checkpoint(model_cache.config, checkpoint_path=model.checkpoint_path,
                                          vocab_path=model.tokenizer_path, use_deepspeed=False)
        model_cache.gpt_cond_latent, model_cache.speaker_embedding = model_cache.model.get_conditioning_latents(
            audio_path=[model.speaker_reference]
        )
        model_cache.model = model_cache.model.cuda()
        models_cache[lang] = model_cache


if __name__ == '__main__':
    initsvr()
    print("loading done")
    import uvicorn
    uvicorn.run(app, host=server_config.server.host, port=server_config.server.port)
