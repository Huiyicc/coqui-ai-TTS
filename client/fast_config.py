import os.path


class ServerConfig:
    def __init__(self, jsonobj: dict):
        self.port = jsonobj.get("port", 18900)
        self.host = jsonobj.get("host", "127.0.0.1")


class ModleConfig:
    def __init__(self, jsonobj: dict):
        self.model_path = jsonobj.get("path", "")
        if os.path.exists(self.model_path) is False:
            raise FileNotFoundError("Model path not found, path: " + self.model_path)
        self.config_path = os.path.join(self.model_path, "config.json")
        self.checkpoint_path = os.path.join(self.model_path, "checkpoint.pth")
        self.tokenizer_path = os.path.join(self.model_path, "vocab.json")
        self.speaker_reference = os.path.join(self.model_path, "SpeakerReference.wav")
        self.speaker_reference_short = os.path.join(self.model_path, "SpeakerReferenceShort.wav")
        self.language = jsonobj.get("lang", "")


class ModleWhisper:
    def __init__(self, jsonobj: dict):
        self.model_path = jsonobj.get("path", "")
        # if os.path.exists(self.model_path) is False:
        #     raise FileNotFoundError("Whisper Model path not found")


class ModelAppConfig:
    def __init__(self, jsonobj: dict):
        self.upload_path = jsonobj.get("upload_path", "upload")


class Config:
    models: dict[str, ModleConfig] = {}
    model_whisper: ModleWhisper = None
    app_config: ModelAppConfig = None

    def __init__(self, svr: ServerConfig, models: dict[str, ModleConfig], whisper: ModleWhisper = None,
                 app_config: ModelAppConfig = None):
        self.server = svr
        for lang, model in models.items():
            if model.language in self.models.keys():
                raise ValueError("Duplicate language")
            if model.language == "":
                raise ValueError("Language is empty")
            self.models[lang] = model
        self.model_whisper = whisper
        self.app_config = app_config


import json


def init_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        svr = ServerConfig(data.get("server", {}))
        models = {}
        for lang, model in data.get("models", {}).items():
            models[lang] = ModleConfig(model)
        whisper = ModleWhisper(data.get("model_whisper", {}))
        app_config = ModelAppConfig(data.get("app", {}))
    return Config(svr, models, whisper, app_config)
