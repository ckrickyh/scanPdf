import os
import time
import psycopg
from pgvector.psycopg import register_vector
import gradio as gr
from google import genai
from google.genai import types

from chunkingRAG.db_config import (
    get_pg_conn_string,
    get_embedding_model,
    get_ollama_base_url,
    get_db_target,
    get_gemini_api_key,
    get_gemini_model,
)

try:
    from chunkingRAG.reranker import rerank_candidates, RerankerService
except ImportError:
    from reranker import rerank_candidates, RerankerService


# 取得配置參數
TARGET = get_db_target()
CONN_STRING = get_pg_conn_string()
MODEL_NAME = get_embedding_model()
OLLAMA_URL = get_ollama_base_url()
GEMINI_MODEL = get_gemini_model()


def get_query_embedding(query: str) -> list[float]:
    """
    自適應計算問題向量：
    1. 優先嘗試連線 Ollama（本地開發模式）
    2. 若 Ollama 無法連線（如部署於雲端/HF Spaces），降級至 SentenceTransformer 本地推論
    """
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        res = client.embeddings(model=MODEL_NAME, prompt=query)
        return res["embedding"] #傳回向量 e.g. [0.0152, -0.0418, 0.0821, ...]
    except Exception:
        # 降級使用 SentenceTransformer
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(f"BAAI/{MODEL_NAME}" if "/" not in MODEL_NAME else MODEL_NAME)
        return st_model.encode(query).tolist()


def generate_llm_answer(
    query: str,
    chunks_data: list[dict | tuple],
    top_p: float = 0.85,
    temperature: float = 0.2,
    response_style: str = "標準詳細",
) -> str:
    """
    使用 Google Gemini API 依據檢索切塊生成綜合繁體中文回答。
    支援 Top-P、Temperature 與 回覆篇幅深度（Token 限制與語意規範）調優。
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return (
            "> [!NOTE]\n"
            "> **未設定 Google Gemini API 金鑰**\n"
            "> 系統已完成切塊檢索。若需啟用 AI 統整回答，請於專案根目錄 `.env` 填入 `GEMINI_API_KEY`。"
        )

    if not chunks_data:
        return "資料庫查無相關切塊，無法提供 AI 綜合回答。"

    # 篇幅與深度配置
    style_configs = {
        "精簡條列": {
            "max_tokens": 400,
            "instruction": "請嚴格將回覆精簡在 150 字以內，僅以極簡重點項目（Bullet points）列出核心解答，省略冗長背景鋪陳與過渡文字。"
        },
        "標準詳細": {
            "max_tokens": 1000,
            "instruction": "請以結構化段落完整作答，兼顧技術背景、操作步驟與核心事實依據。"
        },
        "深入完整": {
            "max_tokens": 2500,
            "instruction": "請進行深度技術解析，詳盡列出所有關聯之數據規格、操作條件、原理細節與潛在防護注意事項。"
        }
    }
    current_style = style_configs.get(response_style, style_configs["標準詳細"])

    # 組合上下文與來源資訊
    context_blocks = []
    for rank, item in enumerate(chunks_data, start=1):
        if isinstance(item, dict):
            src = item.get("source_file", "")
            chapter = item.get("chapter_path", "")
            ctype = item.get("chunk_type", "")
            score = item.get("cosine_similarity", 0.0)
            rerank_s = item.get("rerank_score")
            content = item.get("content", "")
            rerank_info = f"，Rerank 評分：{rerank_s:.4f}" if rerank_s is not None else ""
        else:
            cid, src, ctype, chapter, content, score = item[:6]
            rerank_info = ""

        chapter_str = f"（章節：{chapter}）" if chapter else ""
        context_blocks.append(
            f"【資料來源 {rank}】檔案：{src}{chapter_str}，類型：{ctype}，相似度：{score:.4f}{rerank_info}\n{content}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""你是一個專業的繁體中文技術手冊問答助手。你的任務是嚴格根據提供的【參考文件】回答使用者的問題。

回答規範：
1. 一律使用台灣繁體中文回覆。
2. 嚴格根據參考文件作答，不得憑空捏造或加入未提及的事實。
3. 若參考文件未包含足夠資訊回答問題，請直接明確告知無法從現有資料獲取答案。
4. 回答請條理清晰、層次分明，可適度標註資訊引用的來源序號（例如：[資料來源 1]）。
5. 篇幅與深度規範：{current_style["instruction"]}

【參考文件】：
{context_text}

【使用者問題】：
{query}

請開始回答："""

    max_retries = 3
    base_delay = 1.5  # 秒
    candidate_models = [GEMINI_MODEL]
    if GEMINI_MODEL != "gemini-2.5-flash-lite":
        candidate_models.append("gemini-2.5-flash-lite")

    last_error = None
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=current_style["max_tokens"],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    for current_model in candidate_models:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    if current_model != GEMINI_MODEL:
                        return (
                            f"> [!NOTE]\n"
                            f"> **主模型（{GEMINI_MODEL}）連線繁忙，已自動降級切換至備援模型（{current_model}）生成回答**\n\n"
                            + response.text
                        )
                    return response.text
                return "模型未回傳有效文字內容。"

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # 判定是否為可重試之暫時性尖峰或配額限制 (429 流量管制, 503 服務超載, 504 逾時, 網路波動)
                is_retryable = any(
                    kw in err_str
                    for kw in ["429", "resource_exhausted", "quota", "503", "unavailable", "timeout", "deadline", "connection", "overloaded"]
                )

                if not is_retryable:
                    # 遇到認證失敗或參數錯誤等非暫時性錯誤，不再重試直接中止
                    break

                if attempt < max_retries:
                    wait_time = base_delay * (2 ** (attempt - 1))
                    print(f"[Gemini 呼叫重試] 模型 {current_model} 第 {attempt} 次呼叫繁忙 ({str(e)})，將於 {wait_time:.1f} 秒後重試...")
                    time.sleep(wait_time)

    return (
        f"> [!WARNING]\n"
        f"> **Google Gemini API 呼叫失敗（已自動重試 {max_retries} 次）**\n"
        f"> 最後錯誤原因：`{str(last_error)}`\n\n"
        f"> 建議：當前 Google API 流量處於尖峰或已達免費層每分鐘呼叫上限（RPM/TPM），請稍候 30 至 60 秒後再次嘗試。"
    )


def get_database_statistics() -> str:
    """取得當前連線資料庫之切塊統計數據與系統狀態"""
    target = get_db_target()
    conn_str = get_pg_conn_string()
    gemini_key = get_gemini_api_key()
    gemini_status = f"已啟用（{GEMINI_MODEL}）" if gemini_key else "未設定（僅檢索）"

    try:
        with psycopg.connect(conn_str, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*), count(embedding) FROM manual_chunks;")
                total, with_emb = cur.fetchone()

                cur.execute("SELECT count(*) FROM manual_chunks WHERE chunk_type = 'table';")
                tables = cur.fetchone()[0]

                cur.execute("SELECT count(*) FROM manual_chunks WHERE chunk_type = 'text';")
                texts = cur.fetchone()[0]

        return (
            f"**連線狀態**：連線正常\n"
            f"- **目標環境**：`{target.upper()}`\n"
            f"- **總區塊數**：`{total}` 筆\n"
            f"- **純文字區塊**：`{texts}` 筆\n"
            f"- **表格區塊**：`{tables}` 筆\n"
            f"- **連線端點**：`{'Supabase Cloud (SSL 加密連線)' if target == 'supabase' else 'Localhost (本機連線)'}`\n"
            f"- **Gemini 狀態**：`{gemini_status}`"
        )
    except Exception as e:
        return (
            f"**連線狀態**：連線異常\n"
            f"- **目標環境**：`{target.upper()}`\n"
            f"- **原因**：{str(e)}\n"
            f"- **Gemini 狀態**：`{gemini_status}`"
        )

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    gpu_decorator = lambda f: f
    
@gpu_decorator
def perform_search(
    query: str,
    chunk_type: str,
    top_k: int,
    use_rerank: bool = True,
    candidates_k: int = 15,
    top_p: float = 0.85,
    use_hybrid: bool = True,
    response_style: str = "標準詳細",
) -> tuple[str, str]:
    """執行雙路混合檢索（Vector + BM25 tsvector）、Cross-Encoder 重排與 Google Gemini 生成回答"""
    if not query.strip():
        return "請輸入有效查詢內容。", "無檢索結果。"

    try:
        query_vec = get_query_embedding(query)
    except Exception as e:
        err_msg = f"計算嵌入向量失敗：{str(e)}"
        return err_msg, err_msg

    fetch_limit = candidates_k if use_rerank else top_k
    conn_str = get_pg_conn_string()

    try:
        with psycopg.connect(conn_str, connect_timeout=5) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                if use_hybrid:
                    type_filter_vec = "AND chunk_type = %(chunk_type)s" if chunk_type != "全部" else ""
                    type_filter_kw = "AND chunk_type = %(chunk_type)s" if chunk_type != "全部" else ""

                    sql = f"""
                        WITH vector_search AS (
                            SELECT 
                                chunk_id,
                                source_file,
                                chunk_type,
                                chapter_path,
                                content,
                                1 - (embedding <=> %(vec)s::vector) AS cosine_similarity,
                                ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank_vec
                            FROM manual_chunks
                            WHERE 1 - (embedding <=> %(vec)s::vector) >= 0.3
                            {type_filter_vec}
                            LIMIT %(fetch_k)s
                        ),
                        keyword_search AS (
                            SELECT 
                                chunk_id,
                                source_file,
                                chunk_type,
                                chapter_path,
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
                        "fetch_k": max(30, fetch_limit * 2),
                        "limit": fetch_limit,
                    }
                    if chunk_type != "全部":
                        params["chunk_type"] = chunk_type

                    cur.execute(sql, params)
                    raw_results = cur.fetchall()

                else:
                    type_clause = "WHERE chunk_type = %s" if chunk_type != "全部" else ""
                    sql = f"""
                        SELECT 
                            chunk_id,
                            source_file,
                            chunk_type,
                            chapter_path,
                            content,
                            1 - (embedding <=> %s::vector) AS cosine_similarity,
                            NULL AS rank_vec,
                            NULL AS rank_kw,
                            NULL AS rrf_score
                        FROM manual_chunks
                        {type_clause}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """
                    params = (query_vec, chunk_type, query_vec, fetch_limit) if chunk_type != "全部" else (query_vec, query_vec, fetch_limit)
                    cur.execute(sql, params)
                    raw_results = cur.fetchall()

    except Exception as e:
        err_msg = f"資料庫查詢執行錯誤：{str(e)}"
        return err_msg, err_msg

    if not raw_results:
        return "資料庫查無符合條件之切塊，無法產出 AI 回答。", "查無符合條件之切塊。"

    # 1. 兩階段檢索：啟用 Cross-Encoder 深度重排序
    if use_rerank:
        final_chunks = rerank_candidates(query=query, candidates=raw_results, top_n=top_k)
    else:
        final_chunks = [
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
            for r in raw_results[:top_k]
        ]

    # 2. 呼叫 Google Gemini 生成綜合回答（注入 top_p 與回覆長度控制）
    llm_answer = generate_llm_answer(query, final_chunks, top_p=top_p, response_style=response_style)

    # 3. 格式化檢索切塊明細展示
    md_output = []
    search_desc = "雙路混合檢索（Vector + BM25 RRF）" if use_hybrid else "單路向量檢索"
    mode_desc = f"{search_desc} -> Cross-Encoder 重排（初篩召回：{len(raw_results)} 筆 -> 精選：{len(final_chunks)} 筆）" if use_rerank else f"{search_desc}（Top-{top_k}）"
    md_output.append(f"> **檢索模式**：{mode_desc} | **Top-P**：`{top_p}` | **回覆長度**：`{response_style}`\n")

    for rank, item in enumerate(final_chunks, start=1):
        cid = item["chunk_id"]
        src = item["source_file"]
        ctype = item["chunk_type"]
        chapter_display = item["chapter_path"] if item["chapter_path"] else "無章節資訊"
        content = item["content"]
        score = item["cosine_similarity"]
        rerank_score = item.get("rerank_score")
        rrf_s = item.get("rrf_score")

        badges = [f"向量相似度：`{score:.4f}`"]
        if rrf_s is not None:
            v_rank = f"#{item['rank_vec']}" if item.get("rank_vec") else "未入榜"
            k_rank = f"#{item['rank_kw']}" if item.get("rank_kw") else "未入榜"
            badges.append(f"RRF 得分：`{rrf_s:.5f}` (向量: {v_rank}, 關鍵字: {k_rank})")
        if rerank_score is not None:
            badges.append(f"**Rerank 排序分：`{rerank_score:.4f}`**")

        score_badge = " | ".join(badges)

        md_output.append(
            f"### 【第 {rank} 名】{score_badge}\n"
            f"- **區塊識別碼**：`{cid}`\n"
            f"- **檔案來源**：`{src}`\n"
            f"- **章節路徑**：`{chapter_display}`\n"
            f"- **區塊類型**：`{ctype.upper()}`\n\n"
            f"```text\n{content}\n```\n"
            f"---\n"
        )

    chunks_md = "\n".join(md_output)
    return llm_answer, chunks_md



# 建立 Gradio 使用者介面
custom_css = """
/* ============================================================== */
/* 基礎樣式（明亮模式預設） */
/* ============================================================== */
.status-box {
    background-color: #f8fafc;
    color: #1e293b;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #cbd5e1;
    line-height: 1.6;
}
.status-box p, .status-box li, .status-box span, .status-box strong {
    color: #1e293b;
}
.status-box code {
    background-color: #e2e8f0;
    color: #0284c7;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}

.answer-box {
    background-color: #f0fdf4;
    color: #064e3b;
    border-radius: 8px;
    padding: 18px;
    border: 1px solid #86efac;
    min-height: 120px;
    line-height: 1.7;
    font-size: 15px;
}
.answer-box p, .answer-box li, .answer-box span, .answer-box div {
    color: #064e3b;
}
.answer-box strong {
    color: #022c22;
}
.answer-box code {
    background-color: #dcfce7;
    color: #15803d;
    padding: 2px 6px;
    border-radius: 4px;
}

.main-title, .main-title h1 {
    color: #ea580c !important;
    font-weight: 700;
}

/* ============================================================== */
/* 深色模式支援（包含 Gradio .dark 類別與各層級選取器） */
/* ============================================================== */
.dark .status-box,
body.dark .status-box,
gradio-app.dark .status-box,
[data-theme="dark"] .status-box,
:is(.dark, [data-theme="dark"]) .status-box {
    background-color: #1e293b !important;
    border-color: #475569 !important;
}

.dark .status-box,
.dark .status-box *,
body.dark .status-box *,
gradio-app.dark .status-box *,
[data-theme="dark"] .status-box * {
    color: #f8fafc !important;
}

.dark .status-box code,
body.dark .status-box code,
gradio-app.dark .status-box code {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border: 1px solid #334155;
}

.dark .answer-box,
body.dark .answer-box,
gradio-app.dark .answer-box,
[data-theme="dark"] .answer-box,
:is(.dark, [data-theme="dark"]) .answer-box {
    background-color: #064e3b !important;
    border-color: #10b981 !important;
}

.dark .answer-box,
.dark .answer-box *,
body.dark .answer-box *,
gradio-app.dark .answer-box *,
[data-theme="dark"] .answer-box * {
    color: #f0fdf4 !important;
}

.dark .answer-box strong,
body.dark .answer-box strong {
    color: #ffffff !important;
}

.dark .answer-box code,
body.dark .answer-box code {
    background-color: #022c22 !important;
    color: #6ee7b7 !important;
    border: 1px solid #047857;
}

/* ============================================================== */
/* 系統層級深色模式（Mac / 瀏覽器 prefers-color-scheme，無類別亦強制生效） */
/* ============================================================== */
@media (prefers-color-scheme: dark) {
    .status-box {
        background-color: #1e293b !important;
        border-color: #475569 !important;
    }
    .status-box,
    .status-box *,
    .status-box p,
    .status-box li,
    .status-box span {
        color: #f8fafc !important;
    }
    .status-box code {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
    }

    .answer-box {
        background-color: #064e3b !important;
        border-color: #10b981 !important;
    }
    .answer-box,
    .answer-box *,
    .answer-box p,
    .answer-box li,
    .answer-box span,
    .answer-box div {
        color: #f0fdf4 !important;
    }
    .answer-box strong {
        color: #ffffff !important;
    }
    .answer-box code {
        background-color: #022c22 !important;
        color: #6ee7b7 !important;
        border: 1px solid #047857 !important;
    }
}
"""


with gr.Blocks(title="PDF 語意切分檢索系統") as demo:
    gr.Markdown("# <span style='color: #ea580c;'>PDF 語意切分與 RAG 向量問答平台</span>", elem_classes=["main-title"])
    gr.Markdown(
        "本系統支援雙向資料庫切換（本地 PostgreSQL / 線上 Supabase pgvector），"
        "結合 BGE-M3 向量檢索與 Google Gemini LLM 模型，提供技術手冊的精確語意問答。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            status_display = gr.Markdown(
                value="連線狀態：點擊下方按鈕或執行檢索以更新連線資訊。",
                label="資料庫與系統狀態",
                elem_classes=["status-box"]
            )
            refresh_btn = gr.Button("測試連線狀態", size="sm")

            gr.Markdown("### 檢索與提問設定")
            input_query = gr.Textbox(
                label="查詢提問",
                placeholder="例如：手動更換濾網的保養週期為多久？",
                lines=3
            )
            input_type = gr.Radio(
                label="區塊類型篩選",
                choices=["全部", "text", "table"],
                value="全部"
            )
            input_top_k = gr.Slider(
                label="最終精選筆數（Top-K）",
                minimum=1,
                maximum=10,
                value=3,
                step=1
            )
            use_hybrid_cb = gr.Checkbox(
                label="啟用混合檢索 (Hybrid: Vector + BM25)",
                value=True,
                info="結合 pgvector 語意與 tsvector 全文關鍵字，經由 RRF 融合演算法提升召回率"
            )
            use_rerank_cb = gr.Checkbox(
                label="啟用 Reranker (BAAI/bge-reranker-v2-m3)",
                value=True,
                info="以交叉編碼模型二次精篩，大幅提高技術手冊回答精準度"
            )
            candidates_slider = gr.Slider(
                label="初篩候選數（Candidate Pool）",
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                info="第一階段向量召回池大小"
            )
            top_p_slider = gr.Slider(
                label="Top-P 核採樣閥值",
                minimum=0.1,
                maximum=1.0,
                value=0.85,
                step=0.05,
                info="調低更忠於手冊原文、調高更具文字發散性"
            )
            response_style_radio = gr.Radio(
                label="回覆篇幅偏好",
                choices=["精簡條列", "標準詳細", "深入完整"],
                value="標準詳細",
                info="控制 AI 生成內容的篇幅長度與技術細節密度（兼具 Token 上限與語意引導）"
            )
            search_btn = gr.Button("開始檢索與回答", variant="primary")

            gr.Examples(
                examples=[
                    ["感測器如何安裝在樹幹上？", "全部", 3, True, 15, 0.85, True, "標準詳細"],
                    ["安全防護措施注意事項", "全部", 3, True, 15, 0.85, True, "精簡條列"],
                    ["規格參數對照表", "table", 2, True, 10, 0.85, True, "深入完整"],
                ],
                inputs=[input_query, input_type, input_top_k, use_rerank_cb, candidates_slider, top_p_slider, use_hybrid_cb, response_style_radio]
            )

        with gr.Column(scale=2):
            gr.Markdown("### AI 綜合解答（Google Gemini）")
            output_llm = gr.Markdown(
                value="請於左側輸入問題並點擊「開始檢索與回答」以產出答案。",
                elem_classes=["answer-box"]
            )

            with gr.Accordion("檢索參考切塊（Top-K 來源明細）", open=True):
                output_chunks = gr.Markdown(
                    value="檢索到的參考切塊將展示於此處。"
                )

    search_btn.click(
        fn=perform_search,
        inputs=[input_query, input_type, input_top_k, use_rerank_cb, candidates_slider, top_p_slider, use_hybrid_cb, response_style_radio],
        outputs=[output_llm, output_chunks]
    )


    refresh_btn.click(
        fn=get_database_statistics,
        inputs=[],
        outputs=status_display
    )

if __name__ == "__main__":
    import threading
    share_enabled = os.getenv("GRADIO_SHARE", "false").strip().lower() == "true"
    # 於背景執行緒預載 Reranker 模型與 FP16 加速，消除首次查詢之 15 秒冷啟動延遲
    threading.Thread(target=RerankerService.get_model, daemon=True).start()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_enabled,
        css=custom_css,
    )


