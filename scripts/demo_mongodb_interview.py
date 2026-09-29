"""
MongoDB (NoSQL) 求職面試與技術展示現場示範腳本
執行方式：
  export PATH="$HOME/.local/bin:$PATH"
  uv run python scripts/demo_mongodb_interview.py
"""

import sys
import os
import time
from datetime import datetime, timezone

# 確保專案根目錄在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nosql.mongo_client import get_mongo_db, init_indexes
from nosql.session_logger import log_chat_turn, log_search_audit
from nosql.analytics_pipeline import (
    get_latency_analytics,
    get_top_queries,
    get_session_summary,
    get_feedback_analytics
)


def run_demo():
    print("=" * 70)
    print("【RAG 系統之 NoSQL (MongoDB) 雙引擎架構求職實戰展示】")
    print("=" * 70)

    # 1. 驗證資料庫連線
    print("\n[步驟 1] 檢測 MongoDB 連線與健康度...")
    db = get_mongo_db()
    if db is None:
        print("[失敗] 無法連接 MongoDB。請確認本機已啟動 MongoDB，或於 .env 設定有效的 MONGODB_URI。")
        print("提示：若使用本機 Docker，可執行：docker run -d -p 27017:27017 --name mongo-demo mongo:7")
        return

    print(f"成功連接目標資料庫：{db.name}")

    # 2. 初始化並檢查索引
    print("\n[步驟 2] 自動化索引自癒與健康檢查（複合索引 + 30 天 TTL 索引）...")
    idx_res = init_indexes()
    print(f"索引配置狀態：{idx_res.get('status')}")
    for coll_name, idx_list in idx_res.get("indexes", {}).items():
        print(f"  * 集合 [{coll_name}] 現有索引：{idx_list}")

    # 3. 模擬高頻會話寫入（展示內嵌模型與 $slice 滑動視窗）
    print("\n[步驟 3] 模擬寫入多輪對話會話（測試單文件原子更新與 50 輪視窗防護）...")
    test_session_id = f"demo_sess_{int(time.time())}"
    user_id = "engineer_candidate"

    mock_queries = [
        ("感測器如何安裝在樹幹上？", "需使用專用金屬束帶固定於 1.5 公尺處。", ["chunk_01", "chunk_02"], 210.5, 5),
        ("電源模組輸入電壓規格為何？", "支援 DC 9V 至 36V 寬電壓輸入。", ["chunk_08"], 185.2, 4),
        ("通訊中斷時如何排查？", "請依序檢查天線接口阻抗與 SIM 卡狀態。", ["chunk_15", "chunk_19"], 340.0, 5)
    ]

    for i, (q, a, chunks, lat, rating) in enumerate(mock_queries, 1):
        success = log_chat_turn(
            session_id=test_session_id,
            user_id=user_id,
            query=q,
            response=a,
            retrieved_chunk_ids=chunks,
            latency_ms=lat,
            turn_id=i,
            feedback={"rating": rating, "comment": "回答精準"}
        )
        print(f"  * 寫入對話輪次 #{i} [{q[:12]}...] -> {'成功' if success else '失敗'}")

    # 4. 模擬檢索稽核時序日誌寫入
    print("\n[步驟 4] 模擬寫入各檢索模式之效能指標稽核日誌...")
    audit_samples = [
        ("感測器如何安裝在樹幹上？", "hybrid", "text", 45.2, 12.1, 153.2, 210.5, 5, [0.032, 0.028]),
        ("電源模組輸入電壓規格為何？", "hybrid", "table", 40.1, 10.5, 134.6, 185.2, 3, [0.035]),
        ("天線阻抗規格", "vector", None, 55.4, 0.0, 0.0, 55.4, 5, [0.89, 0.85]),
        ("通訊中斷", "keyword", None, 0.0, 15.6, 0.0, 15.6, 2, [0.45])
    ]

    for q, qtype, cfilter, vlat, klat, rlat, tlat, count, scores in audit_samples:
        log_search_audit(
            query=q,
            session_id=test_session_id,
            query_type=qtype,
            chunk_type_filter=cfilter,
            vector_latency_ms=vlat,
            keyword_latency_ms=klat,
            rerank_latency_ms=rlat,
            total_latency_ms=tlat,
            result_count=count,
            top_scores=scores
        )
    print(f"  * 成功寫入 {len(audit_samples)} 筆不同檢索模式之時序稽核紀錄。")

    # 5. 執行面試級聚合管線分析（Aggregation Pipeline）
    print("\n[步驟 5] 執行 MongoDB 聚合管線分析報表...")

    print("\n--- 【報表 A：檢索模式效能與延遲分析（$group + $avg）】---")
    latency_stats = get_latency_analytics()
    for row in latency_stats:
        print(f"  模式: {row['_id']:<8} | 總查詢數: {row['total_queries']:<3} | 平均總耗時: {row['avg_total_ms']} ms (向量: {row['avg_vector_ms']}ms, 全文: {row['avg_keyword_ms']}ms, 重排: {row['avg_rerank_ms']}ms)")

    print("\n--- 【報表 B：熱門搜尋詞彙頻次統計（$group + $sort + $limit）】---")
    top_q = get_top_queries(limit=5)
    for row in top_q:
        print(f"  問題:「{row['_id']}」 | 查詢頻次: {row['frequency']} 次 | 平均總延遲: {row['avg_latency_ms']} ms")

    print("\n--- 【報表 C：使用者滿意度評分分佈（$unwind + $match）】---")
    fb = get_feedback_analytics()
    print(f"  全域平均滿意度評分: {fb['avg_rating']} / 5.0 (總回饋數: {fb['total_feedbacks']} 筆)")
    for d in fb["distribution"]:
        print(f"    * {d['_id']} 星好評: {d['count']} 筆")

    # 6. M0 免費版容量監控
    print("\n[步驟 6] 監控 M0 儲存容量與索引健康度（512 MB 配額檢查）...")
    try:
        stats = db.command("dbstats")
        data_mb = stats.get("dataSize", 0) / (1024 * 1024)
        storage_mb = stats.get("storageSize", 0) / (1024 * 1024)
        index_mb = stats.get("indexSize", 0) / (1024 * 1024)
        total_mb = storage_mb + index_mb
        print(f"  * 淨資料容量: {data_mb:.3f} MB")
        print(f"  * 索引開銷: {index_mb:.3f} MB")
        print(f"  * 總磁碟佔用: {total_mb:.3f} MB / 512 MB (安全使用率: {(total_mb / 512) * 100:.2f}%)")
    except Exception as e:
        print(f"  無法讀取 dbstats：{e}")

    print("\n" + "=" * 70)
    print("【示範完成】雙資料庫架構與 MongoDB 聚合管線實證運作正常！")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
