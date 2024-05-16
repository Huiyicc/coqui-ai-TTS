# import torch
# from TTS.api import TTS
#
# # Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Init TTS
# tts = TTS(model_path=r"F:\Engcode\AIAssistant\tts\recipes\ljspeech\xtts_v2\run\training\GPT_XTTS_v2.0_LJSpeech_FT-May-11-2024_04+05PM-dbf1a08a",
#           config_path="F:/Engcode/AIAssistant/tts/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-May-11-2024_04+05PM-dbf1a08a/config.json"
#
#           ).to(device)
#
# # Run TTS
# # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# # Text to speech list of amplitude values as output
# # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# # Text to speech to a file
import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config_path = "F:/Engcode/AIAssistant/tts/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-May-14-2024_09+42AM-dbf1a08a/config.json"
XTTS_CHECKPOINT = r"F:\Engcode\AIAssistant\tts\recipes\ljspeech\xtts_v2\run\training\GPT_XTTS_v2.0_LJSpeech_FT-May-14-2024_09+42AM-dbf1a08a\checkpoint_50000.pth"
TOKENIZER_PATH = r"F:\Engcode\AIAssistant\tts\recipes\ljspeech\xtts_v2\run\training\XTTS_v2.0_original_model_files\vocab.json"
SPEAKER_REFERENCE = r"F:\Engcode\AIAssistant\dataset\LJSpeech-1.1\wavs\LJ001-0002.wav"

ts = [
    "  .ok...",
    "  .help...",
    # "hello guys, are you on duty, need any help?",
    # "Munich, Germany’s third-largest metropolis, is a city of tankards and tech, artworks and eccentricity.",
    # "It's generally a safe place to visit, and few travelers have problems.",
    # "While you clan get by using common sense and street smarts, following our tips will make your trip to Munich go even more smoothly.",
    # "Bring euros in cash,",
    # "Although many places in Munich accept cards, smaller stores, market stalls and local food shops such as bakeries and butchers still run on cash.",
    # "You may also encounter coin-only ticket machines on public transport.",
    # "To avoid having to make a hasty tram exit or missing out on an irresistible baked good, it’s best to have some euros in your pocket at all times."
]

#
# for i, t in enumerate(ts):
#     wav = tts.tts(text=t, speaker_wav=r"F:\Engcode\AIAssistant\dataset\LJSpeech-1.1\wavs\LJ001-0003.wav", language="en")
#     print(f"Saving output{i}.wav")
#     tts.save_wav(wav, f"output{i}.wav")


print("Loading model...")
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

# print("Inference...")
# for i, t in enumerate(ts):
#     out = model.inference(
#         t,
#         "en",
#         gpt_cond_latent,
#         speaker_embedding,
#         speed=1.0,
#         temperature=0.5,  # Add custom parameters here
#     )
#     torchaudio.save(f"output{i}.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
#     print(f"Saved {t} -> output{i}.wav")
nn = 1.5
while True:
    txt = input("Enter text: ")
    if txt[0] == "/":
        nn = float(txt[1:])
    rl = False
    txt = "." + txt
    n = 1
    while len(txt) < 10:
        txt = txt + "... " + txt + "... "
        n += nn
    out = model.inference(
        txt,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=0.95,
        temperature=0.7,
    )
    out["wav"] = out["wav"][:int(len(out["wav"]) // n)]
    torchaudio.save(f"output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
