from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    GCV_AUTH: dict
    SER_MODEL: str
    TOKENIZER: str
    RE_MODEL: str
