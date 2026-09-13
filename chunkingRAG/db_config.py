import os
from pathlib import Path
from dotenv import load_dotenv

# 自動尋找專案根目錄之 .env 檔案並載入
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)


def get_db_target() -> str:
    """取得當前指定的資料庫目標 ('local' 或 'supabase')"""
    return os.getenv("DB_TARGET", "local").strip().lower()


def get_pg_conn_string() -> str:
    """
    動態取得 PostgreSQL 連線字串。
    優先順序：
    1. 系統環境變數中明確設定的 PG_CONN_STRING
    2. 依據 DB_TARGET ('local' 或 'supabase') 讀取對應設定
    """
    explicit_conn = os.getenv("PG_CONN_STRING")
    if explicit_conn:
        return explicit_conn.strip()

    target = get_db_target()
    if target == "supabase":
        conn_str = os.getenv("SUPABASE_PG_CONN_STRING")
        if not conn_str:
            raise ValueError(
                "目前 DB_TARGET 設定為 'supabase'，但 .env 中未找到 SUPABASE_PG_CONN_STRING。"
                "請於 .env 填入 Supabase 連線字串（例如：postgresql://postgres.xxx:password@aws-0-xxx.pooler.supabase.com:5432/postgres?sslmode=require）。"
            )
        return conn_str.strip()

    # 預設為 local 本地資料庫
    local_conn = os.getenv(
        "LOCAL_PG_CONN_STRING",
        "postgresql://postgres:postgres@localhost:5432/scan_pdf_rag"
    )
    return local_conn.strip()


def get_embedding_model() -> str:
    """取得嵌入模型名稱"""
    return os.getenv("EMBEDDING_MODEL", "bge-m3").strip()


def get_ollama_base_url() -> str:
    """取得 Ollama API 服務網址"""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()


def get_gemini_api_key() -> str | None:
    """取得 Google Gemini API 金鑰（支援 GEMINI_API_KEY 或 GOOGLE_API_KEY）"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()
    return None


def get_gemini_model() -> str:
    """取得 Google Gemini 模型名稱，預設為 gemini-2.5-flash"""
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

