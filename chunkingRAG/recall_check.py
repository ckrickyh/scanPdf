#!/usr/bin/env python3
"""技術手冊 RAG 檢索評估工具（Recall & Precision Checker）

本模組提供針對 PostgreSQL + pgvector 向量檢索模組之全自動指標評估，包含：
- Recall@K（召回率）
- Precision@K（精確率）
- Hit Rate@K（命中率）
- MRR@K（平均倒數排名）

支援內建評估基準集與外部 JSON 評估集載入。
執行方式：
    uv run chunkingRAG/recall_check.py
    uv run chunkingRAG/recall_check.py --k-list 1,3,5,10 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import ollama
from pgvector.psycopg import register_vector
import psycopg

try:
    from chunkingRAG.db_config import (
        get_embedding_model,
        get_pg_conn_string,
    )
except ImportError:
    from db_config import (
        get_embedding_model,
        get_pg_conn_string,
    )

PG_CONN_STRING = get_pg_conn_string()
EMBEDDING_MODEL = get_embedding_model()

# ------------------------------------------------------------------------------
# 內建黃金基準評估資料集（Ground Truth Benchmark）
# 涵蓋 PiCUS Q72 技術手冊核心操作、硬體配置、故障排除與表格檢索
# ------------------------------------------------------------------------------
DEFAULT_BENCHMARK_DATASET: list[dict[str, Any]] = [
    {
        "id": "TC-01",
        "query": "感測器如何安裝在樹幹上？",
        "expected_files": ["PiCUSQ72Manual_13.md", "PiCUSQ72Manual_51.md"],
        "required_keywords": ["sensor", "mount", "tree"],
        "expected_type": "text",
        "description": "樹幹感測器安裝步驟與注意事項",
    },
    {
        "id": "TC-02",
        "query": "MP used Transmitter Receiver Histogram",
        "expected_files": ["PiCUSQ72Manual_13.md"],
        "required_keywords": ["transmitter", "receiver", "histogram"],
        "expected_type": "table",
        "description": "發射器與接收器直方圖原生表格檢索",
    },
    {
        "id": "TC-03",
        "query": "No connection to Modules 模組通訊異常排除",
        "expected_files": ["PiCUSQ72Manual_59.md"],
        "required_keywords": ["connection", "module", "cable"],
        "expected_type": "text",
        "description": "模組無法連線之排查指南",
    },
    {
        "id": "TC-04",
        "query": "如何使用皮尺測量樹木幾何形狀與幾何圓周？",
        "expected_files": [
            "PiCUSQ72Manual_14.md",
            "PiCUSQ72Manual_15.md",
            "PiCUSQ72Manual_16.md",
        ],
        "required_keywords": ["geometry", "tree", "measur"],
        "expected_type": "text",
        "description": "樹幹幾何測量皮尺配置",
    },
    {
        "id": "TC-05",
        "query": "PiCUS 儀器電池充電與電源維護規範",
        "expected_files": ["PiCUSQ72Manual_56.md", "PiCUSQ72Manual_61.md"],
        "required_keywords": ["battery", "charge", "power"],
        "expected_type": "text",
        "description": "電池與充電座操作安全指南",
    },
    {
        "id": "TC-06",
        "query": "音波斷層掃描的物理原理與聲速測量",
        "expected_files": ["PiCUSQ72Manual_4.md", "PiCUSQ72Manual_17.md"],
        "required_keywords": ["velocity", "sound", "tomograph"],
        "expected_type": "text",
        "description": "音波斷層掃描核心測量原理",
    },
    {
        "id": "TC-07",
        "query": "如何設定與校準感測器模組地址？",
        "expected_files": ["PiCUSQ72Manual_58.md"],
        "required_keywords": ["module", "address", "program"],
        "expected_type": "text",
        "description": "感測器模組 ID 地址燒錄與設定",
    },
]


def retrieve_candidates(
    query: str,
    top_k: int,
    chunk_type: str | None = None,
) -> list[dict[str, Any]]:
    """向 PostgreSQL 執行 pgvector 餘弦相似度檢索，取得前 top_k 個候選區塊。"""
    res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_vec = res["embedding"]

    type_clause = "WHERE chunk_type = %s" if chunk_type else ""
    sql = f"""
        SELECT 
            chunk_id,
            source_file,
            chunk_type,
            TRIM(BOTH ' > ' FROM (
                COALESCE(h1, '') || 
                CASE WHEN h2 IS NOT NULL AND h2 != '' THEN ' > ' || h2 ELSE '' END || 
                CASE WHEN h3 IS NOT NULL AND h3 != '' THEN ' > ' || h3 ELSE '' END
            )) AS chapter_path,
            content,
            1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM manual_chunks
        {type_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    params = (
        (query_vec, chunk_type, query_vec, top_k)
        if chunk_type
        else (query_vec, query_vec, top_k)
    )

    with psycopg.connect(PG_CONN_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    candidates = []
    for r in rows:
        candidates.append({
            "chunk_id": r[0],
            "source_file": r[1],
            "chunk_type": r[2],
            "chapter": r[3] or "",
            "content": r[4] or "",
            "score": float(r[5]),
        })
    return candidates


def evaluate_candidate_relevance(
    candidate: dict[str, Any],
    test_case: dict[str, Any],
) -> bool:
    """判定單一候選區塊是否命中標準答案。

    判定標準（滿足任一明確條件即算命中）：
    1. chunk_id 明確列於 expected_chunk_ids
    2. source_file 屬於 expected_files 且內容滿足 required_keywords 之語意交集
    3. 若無指定 expected_files，則內容完全涵蓋 required_keywords
    """
    chunk_id = candidate["chunk_id"]
    source_file = candidate["source_file"]
    content_lower = candidate["content"].lower()

    # 1. 精確 ID 比對
    if "expected_chunk_ids" in test_case and test_case["expected_chunk_ids"]:
        if chunk_id in test_case["expected_chunk_ids"]:
            return True

    # 2. 檔案來源比對
    expected_files = test_case.get("expected_files", [])
    required_keywords = [
        k.lower() for k in test_case.get("required_keywords", [])
    ]

    file_matched = (not expected_files) or (source_file in expected_files)

    # 3. 關鍵字覆蓋率（至少命中 2 個或全部）
    if required_keywords:
        keyword_hits = sum(1 for kw in required_keywords if kw in content_lower)
        keyword_matched = keyword_hits >= min(2, len(required_keywords))
    else:
        keyword_matched = True

    return file_matched and keyword_matched


def run_recall_check(
    dataset: list[dict[str, Any]],
    k_list: list[int],
    type_filter: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """針對資料集批次執行檢索並計算各階 K 之 Recall, Precision, Hit Rate, MRR 指標。"""
    max_k = max(k_list)
    detailed_results = []

    # 初始化各 K 指標累加器
    metrics_by_k: dict[int, dict[str, float]] = {
        k: {
            "total_recall": 0.0,
            "total_precision": 0.0,
            "total_hits": 0,
            "total_mrr": 0.0,
        }
        for k in k_list
    }

    num_cases = len(dataset)
    print(f"\n{'='*75}")
    print(
        f"開始執行檢索評估 | 測試案例數：{num_cases} | K 階列表：{k_list}"
    )
    print(f"{'='*75}\n")

    for idx, tc in enumerate(dataset, start=1):
        query = tc["query"]
        case_id = tc.get("id", f"TC-{idx:02d}")
        override_type = tc.get("expected_type") or type_filter

        # 取得最大 K 筆候選結果
        candidates = retrieve_candidates(
            query=query, top_k=max_k, chunk_type=override_type
        )

        # 標註每個候選者是否命中
        relevance_flags = [
            evaluate_candidate_relevance(c, tc) for c in candidates
        ]

        # 估算該問題的標準答案目標數（若無明確給定，預設至少為預期檔案數或 1）
        expected_targets_count = max(
            len(tc.get("expected_chunk_ids", [])),
            len(tc.get("expected_files", [])),
            1,
        )

        case_k_metrics = {}
        for k in k_list:
            top_k_flags = relevance_flags[:k]
            hit_count = sum(top_k_flags)

            # Hit Rate: 前 K 筆只要有命中至少 1 個
            is_hit = 1 if hit_count > 0 else 0

            # Recall@K: 命中數 / 目標數 (上限截斷至 1.0)
            recall = min(1.0, hit_count / expected_targets_count)

            # Precision@K: 命中數 / K
            precision = hit_count / k

            # MRR@K: 第一個命中的排名的倒數 (1/rank)
            mrr = 0.0
            for rank_idx, hit in enumerate(top_k_flags, start=1):
                if hit:
                    mrr = 1.0 / rank_idx
                    break

            metrics_by_k[k]["total_recall"] += recall
            metrics_by_k[k]["total_precision"] += precision
            metrics_by_k[k]["total_hits"] += is_hit
            metrics_by_k[k]["total_mrr"] += mrr

            case_k_metrics[k] = {
                "hits": hit_count,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "hit": is_hit,
            }

        top1_cand = candidates[0] if candidates else None
        top1_status = (
            f"命中 ({top1_cand['source_file']})"
            if relevance_flags and relevance_flags[0]
            else "未中"
        )
        top1_score = top1_cand["score"] if top1_cand else 0.0

        detailed_results.append({
            "case_id": case_id,
            "query": query,
            "top1_score": top1_score,
            "top1_status": top1_status,
            "candidates": candidates,
            "relevance_flags": relevance_flags,
            "k_metrics": case_k_metrics,
        })

        # 輸出單題摘要行
        hit_summary = " | ".join(
            f"R@{k}: {case_k_metrics[k]['recall']:.0%} (P@{k}: {case_k_metrics[k]['precision']:.0%})"
            for k in k_list
        )
        print(
            f"[{case_id}] 『{query}』\n"
            f"       Top-1 相似度: {top1_score:.4f} | 首位: {top1_status} | 指標: {hit_summary}"
        )

        if verbose:
            print("       候選結果明細：")
            for r_idx, (cand, hit) in enumerate(
                zip(candidates, relevance_flags), start=1
            ):
                tag = "[HIT]" if hit else "[MISS]"
                print(
                    f"         {tag} 排名 {r_idx}: {cand['source_file']} ({cand['chunk_id']}) "
                    f"| 分數: {cand['score']:.4f} | 類型: {cand['chunk_type']}"
                )
            print()

    # 計算全資料集平均指標
    summary_metrics = {}
    for k in k_list:
        summary_metrics[k] = {
            "Mean_Recall": metrics_by_k[k]["total_recall"] / num_cases,
            "Mean_Precision": metrics_by_k[k]["total_precision"] / num_cases,
            "Hit_Rate": metrics_by_k[k]["total_hits"] / num_cases,
            "MRR": metrics_by_k[k]["total_mrr"] / num_cases,
        }

    # 輸出格式化總結報告
    print(f"\n{'='*75}")
    print("檢索評估總結統計表（Evaluation Summary Table）")
    print(f"{'='*75}")
    header = (
        f"{'指標名稱':<12} | "
        + " | ".join(f"{'K=' + str(k):^12}" for k in k_list)
    )
    print(header)
    print("-" * len(header))

    def format_row(name: str, key: str, is_percent: bool = True) -> str:
        row_vals = []
        for k in k_list:
            v = summary_metrics[k][key]
            row_vals.append(f"{v:^12.2%}" if is_percent else f"{v:^12.4f}")
        return f"{name:<12} | " + " | ".join(row_vals)

    print(format_row("Mean Recall", "Mean_Recall", is_percent=True))
    print(format_row("Mean Precision", "Mean_Precision", is_percent=True))
    print(format_row("Hit Rate", "Hit_Rate", is_percent=True))
    print(format_row("MRR", "MRR", is_percent=False))
    print(f"{'='*75}\n")

    return {
        "summary": summary_metrics,
        "details": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="技術手冊 RAG 向量檢索召回率（Recall）與指標評估工具"
    )
    parser.add_argument(
        "--k-list",
        "-k",
        default="1,3,5,10",
        help="評估之 K 階列表，以逗號分隔（預設：1,3,5,10）",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="外部黃金評估資料集 JSON 檔案路徑",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["text", "table"],
        default=None,
        help="全域篩選區塊類型（text 或 table）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="是否輸出每個候選區塊的詳細比對資訊",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default=None,
        help="將評估結果輸出為 JSON 報告之檔案路徑",
    )

    args = parser.parse_args()

    # 解析 K 列表
    try:
        k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
        k_list = sorted(list(set(k_list)))
    except ValueError:
        print(f"錯誤：--k-list 格式無效：{args.k_list}，必須為整數逗號分隔")
        sys.exit(1)

    # 載入資料集
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"錯誤：指定的評估集檔案不存在：{dataset_path}")
            sys.exit(1)
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功載入外部資料集：{dataset_path}（共 {len(dataset)} 題）")
    else:
        dataset = DEFAULT_BENCHMARK_DATASET
        print(
            f"使用內建 PiCUS Q72 技術手冊基準評估集（共 {len(dataset)} 題）"
        )

    # 執行評估
    report = run_recall_check(
        dataset=dataset,
        k_list=k_list,
        type_filter=args.type,
        verbose=args.verbose,
    )

    # 匯出報告
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        # 移除 candidates 內大文本以精簡輸出
        export_data = {
            "summary": {str(k): v for k, v in report["summary"].items()},
            "details": [
                {
                    "case_id": d["case_id"],
                    "query": d["query"],
                    "top1_score": d["top1_score"],
                    "top1_status": d["top1_status"],
                    "k_metrics": {str(k): v for k, v in d["k_metrics"].items()},
                }
                for d in report["details"]
            ],
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"評估報表已成功匯出至：{export_path.resolve()}")


if __name__ == "__main__":
    main()
