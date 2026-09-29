import logging
from typing import Dict, Any, List
from .mongo_client import get_mongo_db

logger = logging.getLogger("nosql.analytics_pipeline")


def get_latency_analytics() -> List[Dict[str, Any]]:
    """
    聚合管線一：統計各檢索模式（Hybrid / Vector / Keyword）之平均與最大延遲。
    面試亮點：展示 $group, $avg, $max, $round 等標準聚合管道階段。
    """
    db = get_mongo_db()
    if db is None:
        return []

    pipeline = [
        {
            "$group": {
                "_id": "$query_type",
                "total_queries": {"$sum": 1},
                "avg_total_ms": {"$avg": "$total_latency_ms"},
                "avg_vector_ms": {"$avg": "$vector_latency_ms"},
                "avg_keyword_ms": {"$avg": "$keyword_latency_ms"},
                "avg_rerank_ms": {"$avg": "$rerank_latency_ms"},
                "max_total_ms": {"$max": "$total_latency_ms"},
                "min_total_ms": {"$min": "$total_latency_ms"}
            }
        },
        {
            "$project": {
                "total_queries": 1,
                "avg_total_ms": {"$round": ["$avg_total_ms", 2]},
                "avg_vector_ms": {"$round": ["$avg_vector_ms", 2]},
                "avg_keyword_ms": {"$round": ["$avg_keyword_ms", 2]},
                "avg_rerank_ms": {"$round": ["$avg_rerank_ms", 2]},
                "max_total_ms": 1,
                "min_total_ms": 1
            }
        },
        {"$sort": {"total_queries": -1}}
    ]

    try:
        return list(db["search_audit_logs"].aggregate(pipeline))
    except Exception as e:
        logger.error(f"執行延遲聚合管線失敗：{e}")
        return []


def get_top_queries(limit: int = 10) -> List[Dict[str, Any]]:
    """
    聚合管線二：統計最高頻搜尋之問題詞彙及其平均延遲與召回數。
    面試亮點：展示資料流分組排序與熱點詞發掘能力。
    """
    db = get_mongo_db()
    if db is None:
        return []

    pipeline = [
        {
            "$group": {
                "_id": "$query",
                "frequency": {"$sum": 1},
                "avg_latency_ms": {"$avg": "$total_latency_ms"},
                "avg_result_count": {"$avg": "$result_count"}
            }
        },
        {
            "$project": {
                "frequency": 1,
                "avg_latency_ms": {"$round": ["$avg_latency_ms", 2]},
                "avg_result_count": {"$round": ["$avg_result_count", 1]}
            }
        },
        {"$sort": {"frequency": -1}},
        {"$limit": limit}
    ]

    try:
        return list(db["search_audit_logs"].aggregate(pipeline))
    except Exception as e:
        logger.error(f"執行高頻詞聚合管線失敗：{e}")
        return []


def get_session_summary(limit: int = 20) -> List[Dict[str, Any]]:
    """
    聚合管線三：展開會話輪次並計算每個使用者的活躍度與平均響應速度。
    面試亮點：展示 $unwind 拆解內嵌陣列與 $project 重塑結構。
    """
    db = get_mongo_db()
    if db is None:
        return []

    pipeline = [
        {
            "$project": {
                "session_id": 1,
                "user_id": 1,
                "turn_count": 1,
                "created_at": 1,
                "updated_at": 1,
                "avg_turn_latency_ms": {
                    "$round": [{"$avg": "$turns.latency_ms"}, 2]
                }
            }
        },
        {"$sort": {"updated_at": -1}},
        {"$limit": limit}
    ]

    try:
        return list(db["chat_sessions"].aggregate(pipeline))
    except Exception as e:
        logger.error(f"執行會話摘要聚合管線失敗：{e}")
        return []


def get_feedback_analytics() -> Dict[str, Any]:
    """
    聚合管線四：展開 turns 陣列統計使用者滿意度評分分佈。
    面試亮點：展示 $unwind 搭配條件篩選 $match。
    """
    db = get_mongo_db()
    if db is None:
        return {"avg_rating": 0.0, "total_feedbacks": 0, "distribution": []}

    pipeline = [
        {"$unwind": "$turns"},
        {"$match": {"turns.feedback": {"$ne": None}}},
        {
            "$group": {
                "_id": "$turns.feedback.rating",
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]

    try:
        dist = list(db["chat_sessions"].aggregate(pipeline))
        total = sum(item["count"] for item in dist)
        weighted_sum = sum(item["_id"] * item["count"] for item in dist)
        avg = round(weighted_sum / total, 2) if total > 0 else 0.0
        return {
            "avg_rating": avg,
            "total_feedbacks": total,
            "distribution": dist
        }
    except Exception as e:
        logger.error(f"執行滿意度聚合管線失敗：{e}")
        return {"avg_rating": 0.0, "total_feedbacks": 0, "distribution": []}
