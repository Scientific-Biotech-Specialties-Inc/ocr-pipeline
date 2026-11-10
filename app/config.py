from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ocr-pipeline"
    app_env: str = "development"
    app_port: int = 8010

    # API Keys for incoming requests
    serms_api_key: str = "dev-serms-api-key"
    prs_api_key: str = "dev-prs-api-key"
    
    # API Key for callbacks
    callback_api_key: str = "dev-callback-api-key"
    callback_max_retries: int = 3
    callback_backoff_seconds: str = "10,30,60"

    # LLM config (Google AI Studio Gemini)
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout: float = 15.0
    # Line items LLM normalization toggle (enabled by default with fast Gemini inference)
    enable_llm_item_normalization: bool = True

    # Cache TTLs (seconds)
    file_cache_byte_ttl: int = 600          # 10 minutes ephemeral byte handoff
    file_cache_result_ttl: int = 7776000     # 90 days file-hash result cache

    # Databases
    redis_url: str = "redis://ocr_redis:6379/0"
    mongodb_url: str = "mongodb://ocr_mongo:27017"
    mongodb_database: str = "ocr_pipeline"
    postgres_url: str = "postgresql+asyncpg://ocr_user:secret@ocr_postgres:5432/ocr_pipeline"

    # Duplicate detection config
    embedding_model: str = "all-MiniLM-L6-v2"
    duplicate_similarity_threshold: float = 0.85
    duplicate_days_window: int = 90

    # OCR Config
    ocr_language: str = "eng"

    # Image quality gate settings
    image_quality_blur_threshold: float = 80.0
    image_quality_brightness_floor: float = 40.0
    # Orientation-independent size floors. Thermal receipts are often narrow
    # (short side ~200-300px) but still OCR cleanly; require both sides.
    image_quality_min_short_side: int = 200
    image_quality_min_long_side: int = 400
    # Legacy aliases kept so existing IMAGE_QUALITY_MIN_WIDTH/HEIGHT env vars
    # still load without error (values are ignored; short/long sides are used).
    image_quality_min_width: int = 200
    image_quality_min_height: int = 400
    image_quality_max_segments: int = 6
    image_quality_check_not_receipt: bool = True
    image_quality_max_face_area_ratio: float = 0.04
    image_quality_max_saturation_mean: float = 110.0
    image_quality_min_text_edge_density: float = 0.015

    # Long receipt settings
    ocr_max_items: int = 250
    ocr_virtual_tiling: bool = True

    # Rate limiting config
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_ocr_per_minute: int = 30
    rate_limit_duplicate_per_minute: int = 60
    rate_limit_read_per_minute: int = 120
    rate_limit_health_per_minute: int = 60
    rate_limit_auth_fail_per_minute: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
