from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nc_wer_host: str = "0.0.0.0"
    nc_wer_port: int = 8080
    nc_wer_data_dir: str = "./data"
    nc_wer_database_url: str = "sqlite+aiosqlite:///./data/nc_wer.db"
    nc_wer_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    livekit_worker_root: str = ""

    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    hecttor_api_key: str = ""
    sanas_endpoint: str = ""
    sanas_account_id: str = ""
    sanas_account_secret: str = ""
    sanas_secure_media: bool = True

    default_stt_provider: str = "deepgram"
    default_stt_model: str = "nova-2"
    default_stt_language: str = "hi"
    default_turn_align: str = "vad"
    default_nc_strength: float = 0.5

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.nc_wer_cors_origins.split(",") if o.strip()]

    @property
    def data_dir(self):
        from pathlib import Path

        return Path(self.nc_wer_data_dir).resolve()

    @property
    def cache_dir(self):
        return self.data_dir / "cache"

    @property
    def runs_dir(self):
        return self.data_dir / "runs"

    @property
    def uploads_dir(self):
        return self.data_dir / "uploads"


settings = Settings()
