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
    hecttor_api_key: str = ""
    sarvam_api_key: str = ""
    google_service_account_json: str = ""
    google_cloud_project: str = ""
    google_stt_location: str = "us"

    default_stt_provider: str = "deepgram"
    default_stt_model: str = "nova-2"
    default_stt_language: str = "hi"
    default_turn_align: str = "forced"
    # Stacked scoring: forced align + ITN + OI-WER lattice (cross-script fuzzy
    # matching is applied inside both explain_wer and explain_oiwer).
    default_scoring: str = "itn+oiwer"
    default_nc_strength: float = 0.5
    default_hush_strength: float = 0.35

    # Execution mode: "server" = multiprocessing (ProcessPool) + asyncio fan-out;
    # "local" = single-process sequential across calls (asyncio STT overlap only).
    default_execution_mode: str = "server"

    # Parallelism: process workers ≈ vCPUs/2 on n2-standard-16 → 8
    worker_processes: int = 8
    worker_cpu_threads: int = 2
    stt_async_concurrency: int = 4

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
