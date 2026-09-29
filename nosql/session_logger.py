import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, List
from .mongo_client import get_mongo_db
from .models import ChatTurnModel, SearchAuditLogModel, FeedbackModel

logger = logging.getLogger("nosql.session_logger")

# 使用雙工作執行緒之背景執行緒池，確保不佔用 Gradio 主事件迴圈
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mongo_logger")


def log_chat_turn(
    session_id: str,
    user_id: str,
    query: str,
    response: str,
    retrieved_chunk_ids: List[str],
    latency_ms: float,
    turn_id: Optional[int] = None,
    feedback: Optional[dict] = None
) -> bool:
    """
    同步寫入單輪問答至 MongoDB chat_sessions。
    實作滑動視窗防護（最多保留 50 輪問答）與單文件原子操作。
    """
    db = get_mongo_db()
    if db is None:
        return False

    try:
        now = datetime.now(timezone.utc)
        feedback_obj = FeedbackModel(**feedback) if feedback else None

        # 若未指定輪次編號，先預設為 1，底層以 $inc 維護真實 count
        turn_data = ChatTurnModel(
            turn_id=turn_id or 1,
            query=query,
            response=response,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
            feedback=feedback_obj,
            timestamp=now
        ).model_dump(mode="python")

        sessions = db["chat_sessions"]

        # 單文件高內聚更新：
        # 1. $setOnInsert: 首次建立時初始化 session_id, user_id, created_at
        # 2. $push: 將該輪對話推入陣列尾端，以 $slice: -50 限制最多保留最新 50 筆
        # 3. $inc: turn_count 自動累加 1
        # 4. $set: 更新 updated_at
        result = sessions.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "created_at": now
                },
                "$push": {
                    "turns": {
                        "$each": [turn_data],
                        "$slice": -50  # 防禦架構：嚴格限制最多保留 50 輪，杜絕 16 MB 上限溢出
                    }
                },
                "$inc": {"turn_count": 1},
                "$set": {"updated_at": now}
            },
            upsert=True
        )
        return result.acknowledged
    except Exception as e:
        logger.warning(f"寫入 MongoDB 對話會話失敗（不中斷主程式）：{e}")
        return False


def log_search_audit(
    query: str,
    session_id: Optional[str] = None,
    query_type: str = "hybrid",
    chunk_type_filter: Optional[str] = None,
    vector_latency_ms: float = 0.0,
    keyword_latency_ms: float = 0.0,
    rerank_latency_ms: float = 0.0,
    total_latency_ms: float = 0.0,
    result_count: int = 0,
    top_scores: Optional[List[float]] = None,
    status: str = "success",
    error_message: Optional[str] = None
) -> bool:
    """
    同步寫入檢索稽核時序日誌至 MongoDB search_audit_logs。
    配合 TTL 索引實現自動物理過期清除。
    """
    db = get_mongo_db()
    if db is None:
        return False

    try:
        log_data = SearchAuditLogModel(
            session_id=session_id,
            query=query,
            query_type=query_type,
            chunk_type_filter=chunk_type_filter,
            vector_latency_ms=vector_latency_ms,
            keyword_latency_ms=keyword_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            total_latency_ms=total_latency_ms,
            result_count=result_count,
            top_scores=top_scores or [],
            status=status,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc)
        ).model_dump(mode="python")

        audit_logs = db["search_audit_logs"]
        result = audit_logs.insert_one(log_data)
        return result.acknowledged
    except Exception as e:
        logger.warning(f"寫入 MongoDB 檢索稽核日誌失敗（不中斷主程式）：{e}")
        return False


def log_chat_turn_async(*args, **kwargs):
    """非同步非阻塞排程寫入對話會話"""
    _executor.submit(log_chat_turn, *args, **kwargs)


def log_search_audit_async(*args, **kwargs):
    """非同步非阻塞排程寫入稽核日誌"""
    _executor.submit(log_search_audit, *args, **kwargs)
