import argparse
import os
import psycopg
from pgvector.psycopg import register_vector
import ollama

try:
    from chunkingRAG.db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_db_target,
    )
    from chunkingRAG.reranker import rerank_candidates
except ImportError:
    from db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_db_target,
    )
    from reranker import rerank_candidates

PG_CONN_STRING = get_pg_conn_string()
EMBEDDING_MODEL = get_embedding_model()


def search_manual(
    query: str,
    top_k: int = 3,
    chunk_type: str | None = None,
    min_similarity: float = 0.5,
    use_reranker: bool = False,
    candidates_k: int = 15,
    use_hybrid: bool = True,
):
    """
    執行技術手冊檢索。
    支援：
    1. 雙路混合檢索（Hybrid Search）：向量語意（pgvector）+ 全文檢索（tsvector BM25）以 RRF 演算法融合。
    2. 單階段純向量檢索。
    3. 第二階段 Cross-Encoder（bge-reranker-v2-m3）深度語意重排序。
    """
    # 1. 計算問題向量
    res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_vec = res["embedding"]

    limit_k = candidates_k if use_reranker else top_k

    with psycopg.connect(PG_CONN_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if use_hybrid:
                # 雙路並行檢索 + RRF（Reciprocal Rank Fusion）融合
                type_filter_vec = "AND chunk_type = %(chunk_type)s" if chunk_type else ""
                type_filter_kw = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

                sql = f"""
                    WITH vector_search AS (
                        SELECT 
                            chunk_id,
                            source_file,
                            chunk_type,
                            concat_ws(' > ', NULLIF(h1, ''), NULLIF(h2, ''), NULLIF(h3, ''), NULLIF(h4, ''), NULLIF(h5, '')) AS chapter_path,
                            content,
                            1 - (embedding <=> %(vec)s::vector) AS cosine_similarity,
                            ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank_vec
                        FROM manual_chunks
                        WHERE 1 - (embedding <=> %(vec)s::vector) >= %(min_sim)s
                        {type_filter_vec}
                        LIMIT %(fetch_k)s
                    ),
                    keyword_search AS (
                        SELECT 
                            chunk_id,
                            source_file,
                            chunk_type,
                            concat_ws(' > ', NULLIF(h1, ''), NULLIF(h2, ''), NULLIF(h3, ''), NULLIF(h4, ''), NULLIF(h5, '')) AS chapter_path,
                            content,
                            1 - (embedding <=> %(vec)s::vector) AS cosine_similarity,
                            ts_rank(tsv, plainto_tsquery('english', %(query)s)) AS bm25_score,
                            ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, plainto_tsquery('english', %(query)s)) DESC) AS rank_kw
                        FROM manual_chunks
                        WHERE tsv @@ plainto_tsquery('english', %(query)s)
                        {type_filter_kw}
                        LIMIT %(fetch_k)s
                    ),
                    fused AS (
                        SELECT 
                            COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
                            COALESCE(v.source_file, k.source_file) AS source_file,
                            COALESCE(v.chunk_type, k.chunk_type) AS chunk_type,
                            COALESCE(v.chapter_path, k.chapter_path) AS chapter_path,
                            COALESCE(v.content, k.content) AS content,
                            COALESCE(v.cosine_similarity, k.cosine_similarity) AS cosine_similarity,
                            v.rank_vec,
                            k.rank_kw,
                            COALESCE(1.0 / (60.0 + v.rank_vec), 0.0) + COALESCE(1.0 / (60.0 + k.rank_kw), 0.0) AS rrf_score
                        FROM vector_search v
                        FULL OUTER JOIN keyword_search k ON v.chunk_id = k.chunk_id
                    )
                    SELECT 
                        chunk_id,
                        source_file,
                        chunk_type,
                        chapter_path,
                        content,
                        cosine_similarity,
                        rank_vec,
                        rank_kw,
                        rrf_score
                    FROM fused
                    ORDER BY rrf_score DESC
                    LIMIT %(limit)s;
                """
                params = {
                    "vec": query_vec,
                    "query": query,
                    "min_sim": min_similarity,
                    "fetch_k": max(30, limit_k * 2),
                    "limit": limit_k,
                }
                if chunk_type:
                    params["chunk_type"] = chunk_type

                cur.execute(sql, params)
                raw_candidates = cur.fetchall()

            else:
                # 單路純向量檢索
                conditions = ["1 - (embedding <=> %(vec)s::vector) >= %(min_sim)s"]
                params = {
                    "vec": query_vec,
                    "min_sim": min_similarity,
                    "limit": limit_k,
                }
                if chunk_type:
                    conditions.append("chunk_type = %(chunk_type)s")
                    params["chunk_type"] = chunk_type

                where_clause = "WHERE " + " AND ".join(conditions)
                sql = f"""
                    SELECT 
                        chunk_id,
                        source_file,
                        chunk_type,
                        concat_ws(' > ', NULLIF(h1, ''), NULLIF(h2, ''), NULLIF(h3, ''), NULLIF(h4, ''), NULLIF(h5, '')) AS chapter_path,
                        content,
                        1 - (embedding <=> %(vec)s::vector) AS cosine_similarity,
                        NULL AS rank_vec,
                        NULL AS rank_kw,
                        NULL AS rrf_score
                    FROM manual_chunks
                    {where_clause}
                    ORDER BY embedding <=> %(vec)s::vector
                    LIMIT %(limit)s;
                """
                cur.execute(sql, params)
                raw_candidates = cur.fetchall()

    # 4. 若啟用 Reranker，執行 Cross-Encoder 深度重排序
    if use_reranker and raw_candidates:
        final_results = rerank_candidates(query=query, candidates=raw_candidates, top_n=top_k)
    else:
        final_results = [
            {
                "chunk_id": r[0],
                "source_file": r[1],
                "chunk_type": r[2],
                "chapter_path": r[3] if r[3] else "",
                "content": r[4],
                "cosine_similarity": float(r[5]) if r[5] is not None else 0.0,
                "rank_vec": r[6],
                "rank_kw": r[7],
                "rrf_score": float(r[8]) if r[8] is not None else None,
                "rerank_score": None,
            }
            for r in raw_candidates[:top_k]
        ]

    print(f"\n{'='*60}")
    print(f"搜尋問題：{query}")
    if chunk_type:
        print(f"篩選類型：{chunk_type}")
    print(f"相似度門檻：>= {min_similarity}")
    search_type_desc = "雙路混合檢索（Vector + BM25 RRF）" if use_hybrid else "單路向量檢索"
    if use_reranker:
        print(f"檢索模式：{search_type_desc} -> Cross-Encoder 重排（初篩召回：{len(raw_candidates)} 筆 -> 重排精選：{len(final_results)} 筆）")
    else:
        print(f"檢索模式：{search_type_desc}（回傳筆數：{len(final_results)} / Top-{top_k}）")
    print(f"{'='*60}")

    for rank, item in enumerate(final_results):
        chapter_display = item["chapter_path"] if item["chapter_path"] else "無章節資訊"
        score_parts = [f"向量相似度：{item['cosine_similarity']:.4f}"]
        if item.get("rrf_score") is not None:
            score_parts.append(f"RRF 得分：{item['rrf_score']:.5f}")
            v_rank = f"#{item['rank_vec']}" if item.get("rank_vec") else "未入榜"
            k_rank = f"#{item['rank_kw']}" if item.get("rank_kw") else "未入榜"
            score_parts.append(f"(向量: {v_rank}, 關鍵字: {k_rank})")
        if item.get("rerank_score") is not None:
            score_parts.append(f"Rerank 評分：{item['rerank_score']:.4f}")

        score_info = " | ".join(score_parts)
        print(f"\n[排名 {rank + 1} | {score_info} | 類型：{item['chunk_type']} | 章節：{chapter_display}]")
        print(f"來源檔案：{item['source_file']} (ID: {item['chunk_id']})")
        content = item["content"]
        preview = content[:250] + ("..." if len(content) > 250 else "")
        print(f"內容摘錄：\n{preview}")
        print("-" * 60)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="技術手冊向量檢索與 Reranker 重排序查詢工具")
    parser.add_argument("--query", "-q", default="感測器如何安裝在樹幹上？", help="搜尋提問文字")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="最終回傳前 K 筆最精準結果")
    parser.add_argument("--candidates", "-c", type=int, default=15, help="第一階段粗篩候選數量（預設：15）")
    parser.add_argument("--rerank", "-r", action="store_true", help="啟用 Cross-Encoder（bge-reranker）重排序")
    parser.add_argument("--no-hybrid", action="store_false", dest="hybrid", default=True, help="停用混合檢索，改為純向量檢索")
    parser.add_argument("--type", "-t", choices=["text", "table"], default=None, help="限定區塊類型（text 或 table）")
    parser.add_argument("--min-sim", "-m", type=float, default=0.5, help="最低餘弦相似度門檻（預設：0.5）")
    args = parser.parse_args()

    search_manual(
        query=args.query,
        top_k=args.top_k,
        chunk_type=args.type,
        min_similarity=args.min_sim,
        use_reranker=args.rerank,
        candidates_k=args.candidates,
        use_hybrid=args.hybrid,
    )


