import os
import logging
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("nosql.mongo_client")

_client_instance: Optional[MongoClient] = None


def get_mongo_client() -> Optional[MongoClient]:
    """
    取得 MongoDB 單例連線物件，針對 Atlas M0 免費版或本機環境調優連線池。
    若連線失敗則回傳 None 並記錄日誌，不中斷主服務。
    """
    global _client_instance

    enabled = os.getenv("MONGODB_ENABLED", "true").lower() in ("true", "1", "yes")
    if not enabled:
        return None

    if _client_instance is not None:
        return _client_instance

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    try:
        _client_instance = MongoClient(
            mongo_uri,
            maxPoolSize=20,            # 嚴格控制連線數，防止觸及 M0 500 連線上限
            minPoolSize=2,             # 基礎連線池保持活躍
            maxIdleTimeMS=30000,       # 30 秒閒置回收
            serverSelectionTimeoutMS=3000, # 3 秒快速失敗，防範連線掛起
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        # 驗證連線健康度
        _client_instance.admin.command("ping")
        logger.info("MongoDB 連線成功建立。")
        return _client_instance
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.warning(f"MongoDB 連線失敗（此狀態不影響 PostgreSQL 核心檢索）：{e}")
        _client_instance = None
        return None


def get_mongo_db() -> Optional[Database]:
    """
    取得專案目標資料庫實體。
    """
    client = get_mongo_client()
    if client is None:
        return None
    db_name = os.getenv("MONGODB_DB_NAME", "scanpdf_analytics")
    return client[db_name]


def init_indexes() -> dict:
    """
    自動化初始化索引（複合索引、唯一鍵、TTL 自動淘汰索引）。
    回傳建立索引之結果狀態字典。
    """
    db = get_mongo_db()
    if db is None:
        return {"status": "skipped", "reason": "MongoDB not connected"}

    results = {}
    try:
        # 1. chat_sessions 集合索引
        sessions = db["chat_sessions"]
        # 唯一索引：session_id
        idx_sess = sessions.create_index([("session_id", ASCENDING)], unique=True)
        # 複合索引：user_id 正序 + updated_at 倒序（符合 ESR 原則）
        idx_user_time = sessions.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        results["chat_sessions"] = [idx_sess, idx_user_time]

        # 2. search_audit_logs 集合索引
        audit_logs = db["search_audit_logs"]
        # 原生 TTL 索引：30 天（2592000 秒）後底層背景自動清除
        idx_ttl = audit_logs.create_index(
            [("timestamp", ASCENDING)],
            expireAfterSeconds=2592000,
            name="idx_ttl_30d"
        )
        # 查詢類型與時間複合索引（供延遲統計分析）
        idx_type_time = audit_logs.create_index([("query_type", ASCENDING), ("timestamp", DESCENDING)])
        results["search_audit_logs"] = [idx_ttl, idx_type_time]

        logger.info(f"MongoDB 索引初始化完成：{results}")
        return {"status": "success", "indexes": results}
    except Exception as e:
        logger.error(f"建立 MongoDB 索引異常：{e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("正在檢測 MongoDB 連線與索引狀態...")
    db = get_mongo_db()
    if db is not None:
        print(f"資料庫連線成功：{db.name}")
        index_res = init_indexes()
        print(f"索引配置結果：{index_res}")
    else:
        print("未檢測到可用的 MongoDB 服務，請確認 MONGODB_URI 設定或本機 Docker 服務。")
