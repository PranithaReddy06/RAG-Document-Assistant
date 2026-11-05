from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_model: str 
    embedding_model: str 
    ollama_base_url: str = "http://localhost:11434"
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()