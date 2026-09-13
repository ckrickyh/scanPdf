# 技術手冊 RAG 結構化切分與 PostgreSQL (pgvector) 入庫實作流程

本文件規範 `output/` 目錄中 Markdown 檔案之後處理管線，包含章節解析、HTML 表格隔離、純文字語意切分、上下文句子重疊，以及最終寫入 PostgreSQL 搭配 pgvector 擴充套件之標準實作程序。

---

## 一、 設計目標與核心策略

1. **結構脈絡保留**：先解析 Markdown 多級標題（`#`、`##`、`###`），將章節層級注入各 Chunk 之欄位與 Metadata，避免碎片文字遺失上位主題脈絡。
2. **表格實體隔離**：針對 PaddleOCR 產出之原生 HTML 表格（`<table...</table>`）實施正規表達式過濾，整張表格作為原子級（Atomic）獨立 Chunk 入庫，嚴禁文字切分器切入破壞結構。
3. **語意動態斷句**：非表格文字採用 LangChain 之 `SemanticChunker`，透過模型（如 `bge-m3`）依話題轉換距離下刀，維持主題高凝聚力。
4. **句子級重疊（Overlap）補償**：在語意切分後的相鄰區塊間，自動補上前一塊末尾的 1 句完整文字，防範代名詞指代斷裂與跨區塊語意脫節。
5. **資料庫層級 HNSW 索引加速**：捨棄第三方專有向量庫，直接使用關聯式 PostgreSQL 資料庫與 `pgvector` 擴充套件，在 1024 維特徵空間建立 HNSW 向量索引，達成毫秒級餘弦相似度檢索。

---

## 二、 四階段處理流程架構

```
[原始 Markdown 檔案 (output/*.md)]
                │
                ▼
【階段一：標題結構預解析】
  - MarkdownHeaderTextSplitter (H1 / H2 / H3)
  - 提取章節路徑並暫存為結構化欄位
                │
                ▼
【階段二：HTML 表格抽離隔離】
  - Regex 辨識 <table(?:\s+[^>]*)?>[\s\S]*?</table>
  - 提取表格作為獨立原子 Chunk（chunk_type: table）
                │
                ▼
【階段三：純文字語意切分與句子重疊】
  - 非表格文字送入 SemanticChunker
  - 依語意轉折切塊
  - 自動向後串接 1 句結尾作為前綴重疊（chunk_type: text）
                │
                ▼
【階段四：封裝、向量化與 PostgreSQL 入庫】
  - 組合 Chunk ID、檔案來源、章節資訊與純文字內容
  - 調用 Ollama bge-m3 計算 1024 維度向量
  - 批次寫入 PostgreSQL 資料表並建立 HNSW 向量索引
```

---

## 三、 資料庫結構定義（DDL）

在 PostgreSQL 中啟用 `pgvector` 擴充套件，並建立專用資料表與 HNSW 索引：

```sql
-- 1. 啟用 pgvector 擴充套件
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 建立技術手冊區塊資料表
CREATE TABLE IF NOT EXISTS manual_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id VARCHAR(120) UNIQUE NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    chunk_type VARCHAR(20) NOT NULL, -- 'text' 或 'table'
    chunk_index INT NOT NULL,
    h1 TEXT,
    h2 TEXT,
    h3 TEXT,
    content TEXT NOT NULL,
    embedding vector(1024) NOT NULL, -- 對應 bge-m3 輸出之 1024 維度
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. 建立 HNSW 向量索引（使用 Cosine 距離運算子）
CREATE INDEX IF NOT EXISTS idx_manual_chunks_hnsw 
ON manual_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. 建立輔助查詢索引
CREATE INDEX IF NOT EXISTS idx_manual_chunks_source ON manual_chunks (source_file);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_type ON manual_chunks (chunk_type);
```

---

## 四、 環境配置與依賴管理規範

依據專案管理規範，統一使用 Python 3.12 基準環境與 `uv` 套件管理工具。

### 1. 依賴安裝

```bash
uv add psycopg[binary] pgvector langchain-community langchain-experimental langchain-text-splitters ollama
```

### 2. 本地 Embedding 模型準備

確保本機 Ollama 服務已啟動並下載指定模型：

```bash
ollama pull bge-m3
```

---

## 五、 核心實作代碼規範

建立獨立處理腳本 `chunkingRAG/pipeline_postgres.py`，完整程式碼結構如下：

```python
from pathlib import Path
import re
import psycopg
from pgvector.psycopg import register_vector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# 1. 資料庫連線字串與設定
# 請依實際環境調整連線資訊（或透過環境變數傳入）
PG_CONN_STRING = "postgresql://postgres:postgres@localhost:5432/scan_pdf_rag"
EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = "http://localhost:11434"

# 嚴格匹配包含屬性與樣式之 HTML table 標籤
HTML_TABLE_REGEX = re.compile(
    r"(<table(?:\s+[^>]*)?>[\s\S]*?</table>)",
    re.IGNORECASE
)

# 2. 標題切分規則配置
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False
)

# 3. 語意切分器初始化
langchain_embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

semantic_splitter = SemanticChunker(
    embeddings=langchain_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)

def chunk_text_with_overlap(text: str, overlap_sentences: int = 1) -> list[str]:
    """
    對純文字進行語意切分，並自動為後續區塊補充前一區塊之末尾句子以形成上下文重疊。
    """
    if not text.strip():
        return []
        
    raw_chunks = semantic_splitter.split_text(text)
    if len(raw_chunks) <= 1:
        return raw_chunks

    final_chunks = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_chunk = raw_chunks[i - 1]
        curr_chunk = raw_chunks[i]
        
        # 依繁體中文全形句號與換行分割提取結尾句
        sentences = [s.strip() for s in re.split(r"[。\n]", prev_chunk) if s.strip()]
        overlap_prefix = "。".join(sentences[-overlap_sentences:]) + "。" if sentences else ""
        
        if overlap_prefix:
            final_chunks.append(f"{overlap_prefix}\n{curr_chunk}")
        else:
            final_chunks.append(curr_chunk)
            
    return final_chunks

def init_database(conn: psycopg.Connection):
    """確保資料表與 pgvector 擴充套件存在"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manual_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_id VARCHAR(120) UNIQUE NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                chunk_type VARCHAR(20) NOT NULL,
                chunk_index INT NOT NULL,
                h1 TEXT,
                h2 TEXT,
                h3 TEXT,
                content TEXT NOT NULL,
                embedding vector(1024) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_manual_chunks_hnsw 
            ON manual_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
    conn.commit()

def run_pipeline(output_dir_str: str = "./output"):
    output_dir = Path(output_dir_str)
    if not output_dir.exists():
        raise FileNotFoundError(f"找不到輸入目錄：{output_dir.resolve()}")

    records_to_insert: list[dict] = []
    md_files = sorted(output_dir.glob("*.md"))
    print(f"發現 {len(md_files)} 份 Markdown 檔案，開始執行結構化切分流程...")

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue

        # 階段一：提取標題與章節階層
        sections = header_splitter.split_text(content)
        chunk_idx = 0

        for sec in sections:
            sec_text = sec.page_content
            meta = sec.metadata
            h1 = meta.get("H1")
            h2 = meta.get("H2")
            h3 = meta.get("H3")

            # 階段二：分離 HTML 表格與非表格文字
            matches = list(HTML_TABLE_REGEX.finditer(sec_text))
            last_idx = 0

            for match in matches:
                start, end = match.span()
                pre_text = sec_text[last_idx:start].strip()

                # 階段三（A）：處理表格前的純文字
                if pre_text:
                    for text_chunk in chunk_text_with_overlap(pre_text, overlap_sentences=1):
                        records_to_insert.append({
                            "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                            "source_file": md_file.name,
                            "chunk_type": "text",
                            "chunk_index": chunk_idx,
                            "h1": h1, "h2": h2, "h3": h3,
                            "content": text_chunk
                        })
                        chunk_idx += 1

                # 階段三（B）：處理表格本體（整張獨立入庫）
                table_html = match.group(1).strip()
                records_to_insert.append({
                    "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                    "source_file": md_file.name,
                    "chunk_type": "table",
                    "chunk_index": chunk_idx,
                    "h1": h1, "h2": h2, "h3": h3,
                    "content": table_html
                })
                chunk_idx += 1

                last_idx = end

            # 階段三（C）：處理剩餘之純文字
            tail_text = sec_text[last_idx:].strip()
            if tail_text:
                for text_chunk in chunk_text_with_overlap(tail_text, overlap_sentences=1):
                    records_to_insert.append({
                        "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                        "source_file": md_file.name,
                        "chunk_type": "text",
                        "chunk_index": chunk_idx,
                        "h1": h1, "h2": h2, "h3": h3,
                        "content": text_chunk
                    })
                    chunk_idx += 1

    print(f"切分處理完成，共生成 {len(records_to_insert)} 個知識區塊。開始執行向量化與 PostgreSQL 入庫...")

    # 階段四：連線資料庫並批次寫入
    with psycopg.connect(PG_CONN_STRING) as conn:
        register_vector(conn)
        init_database(conn)

        batch_size = 32
        for i in range(0, len(records_to_insert), batch_size):
            batch = records_to_insert[i:i + batch_size]
            contents = [item["content"] for item in batch]
            
            # 批次計算 bge-m3 向量
            embeddings_vecs = langchain_embeddings.embed_documents(contents)

            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO manual_chunks (
                        chunk_id, source_file, chunk_type, chunk_index,
                        h1, h2, h3, content, embedding
                    ) VALUES (
                        %(chunk_id)s, %(source_file)s, %(chunk_type)s, %(chunk_index)s,
                        %(h1)s, %(h2)s, %(h3)s, %(content)s, %(embedding)s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        h1 = EXCLUDED.h1,
                        h2 = EXCLUDED.h2,
                        h3 = EXCLUDED.h3;
                """, [
                    {**item, "embedding": vec}
                    for item, vec in zip(batch, embeddings_vecs)
                ])
            conn.commit()
            print(f"進度：已成功寫入 {min(i + batch_size, len(records_to_insert))} / {len(records_to_insert)} 筆至 PostgreSQL")

    print("全量資料寫入 PostgreSQL 完畢。")

if __name__ == "__main__":
    run_pipeline()
```

---

## 六、 向量檢索查詢驗證

透過標準 SQL 搭配餘弦距離運算子（`<=>`），驗證 HNSW 索引之 Top-K 檢索效能：

```python
import psycopg
from pgvector.psycopg import register_vector
import ollama

PG_CONN_STRING = "postgresql://postgres:postgres@localhost:5432/scan_pdf_rag"

def search_manual(query: str, top_k: int = 3):
    # 1. 計算問題向量
    res = ollama.embeddings(model="bge-m3", prompt=query)
    query_vec = res["embedding"]

    # 2. 執行 SQL 餘弦距離排序查詢
    with psycopg.connect(PG_CONN_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    chunk_id,
                    source_file,
                    chunk_type,
                    COALESCE(h1, '') || ' > ' || COALESCE(h2, '') AS chapter_path,
                    content,
                    1 - (embedding <=> %s) AS cosine_similarity
                FROM manual_chunks
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_vec, query_vec, top_k))
            
            results = cur.fetchall()

    print(f"=== 搜尋問題：{query} ===")
    for rank, row in enumerate(results):
        cid, src, ctype, chapter, content, score = row
        print(f"\n[排名 {rank + 1} | 相似度：{score:.4f} | 類型：{ctype} | 章節：{chapter}]")
        print(f"來源檔案：{src} (ID: {cid})")
        print(content[:250] + ("..." if len(content) > 250 else ""))

if __name__ == "__main__":
    search_manual("感測器如何安裝在樹幹上？")
```

---

## 七、 潛在風險與工程防錯措施

| 潛在風險點 | 根本原因（Root Cause） | 建議防護機制（Suggested Fix） |
| :--- | :--- | :--- |
| **資料庫未預裝 pgvector** | 基礎 PostgreSQL 映像檔未包含編譯好的 `vector` 擴充套件，執行 `CREATE EXTENSION` 報錯。 | 使用官方支援向量之映像檔（如 `pgvector/pgvector:pg16`）啟動資料庫服務。 |
| **向量維度不符** | 資料表欄位定義為 `vector(1024)`，若誤用 `embeddinggemma`（768 維）寫入會引發維度衝突。 | 確保 Embedding 模型與 DDL 維度宣告嚴格一致（`bge-m3` 對應 1024 維）。 |
| **HNSW 索引建構耗時過長** | 寫入大批資料時若已預先建立 HNSW 索引，每筆插入皆需更新圖狀節點。 | 資料量若突破數十萬筆，建議先執行批量插入（`COPY` 或 `executemany`），最後再執行 `CREATE INDEX`。 |
| **代名詞跨區塊指代不明** | 切分邊界剛好落在以「該儀器」、「此步驟」開頭之句子，缺少前文主詞。 | 依本流程第三階段強制注入前 1 句之末尾文字（Overlap），確保檢索時保有完整語境。 |
