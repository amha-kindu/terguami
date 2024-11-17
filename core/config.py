from pydantic_settings import BaseSettings
import torch, random


torch.manual_seed(3000)
def get_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        torch.cuda.manual_seed_all(3000)
    return torch.device(device_name)

random.seed(3000)


class Settings(BaseSettings):
    D_MODEL: int = 512
    DEVICE: torch.device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    DFF: int = 2048
    DROPOUT: float = 0.1
    HEADS: int = 16
    N_BLOCKS: int = 6
    SEQ_LEN: int = 52
    SRC_LANG: str = "en"
    TGT_LANG: str = "am"
    VOCAB_SIZE: int = 20000
    PROJECT_NAME: str = "Terguami: English to Amharic Translation API"
    MODEL_PATH: str = "/app/models/weights/tmodel-en-am-v1-20k.pt"
    TOKENIZER_FOLDER: str = "/app/models/tokenizers"
    TOKENIZER_BASENAME: str = "tokenizer-{0}-v3.5-20k.json"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 7000

    class Config:
        env_file = ".env"

settings = Settings()
