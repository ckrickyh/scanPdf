import argparse
import os
from pathlib import Path
import re
import sys
import psycopg
from pgvector.psycopg import register_vector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
import ollama

try:
    from chunkingRAG.db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_ollama_base_url,
        get_db_target,
    )
except ImportError:
    from db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_ollama_base_url,
        get_db_target,
    )

# 1. 資料庫連線字串與設定（支援 .env DB_TARGET 切換與環境變數覆蓋）
PG_CONN_STRING = get_pg_conn_string()
EMBEDDING_MODEL = get_embedding_model()
OLLAMA_BASE_URL = get_ollama_base_url()

# 嚴格匹配包含屬性與樣式之 HTML table 標籤
HTML_TABLE_REGEX = re.compile(
    r"(<table(?:\s+[^>]*)?>[\s\S]*?</table>)",
    re.IGNORECASE
)

# 2. 標題切分規則配置（支援 H1 至 H6）
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
    ("#####", "H5"),
    ("######", "H6"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False
)

def build_breadcrumb_from_stack(stack: dict[int, str], max_len: int = 150) -> str:
    """
    自章節層級堆疊字典安全構建麵包屑路徑，
    以整數序數排序徹底避免字母排序錯亂（如 H10 排在 H2 前面），
    並以 ' / ' 分隔，超出 max_len 時實施防稀釋截斷。
    """
    if not stack:
        return "未分類章節"

    sorted_items = sorted(stack.items(), key=lambda x: x[0])
    titles = [item[1] for item in sorted_items]
    full_path = " / ".join(titles)

    if len(full_path) > max_len and len(titles) > 2:
        return f"{titles[0]} / ... / {titles[-1]}"
    return full_path


def build_breadcrumb_path(meta: dict, max_len: int = 150) -> str:
    """自 metadata 字典提取章節階層並轉為路徑字串"""
    h_items = {}
    for k, v in meta.items():
        m = re.match(r"^H(\d+)$", k)
        if m and v and str(v).strip():
            h_items[int(m.group(1))] = str(v).strip()

    return build_breadcrumb_from_stack(h_items, max_len)

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

# 4. 遞迴字元安全兜底切分器（防止語意切分器產出超大區塊導致向量失真或溢出）
safety_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "！", "？", ".", "；", ";", " ", ""]
)


def extract_overlap_prefix(
    prev_chunk: str, target_overlap_chars: int = 250
) -> str:
    """
    自前一區塊提取結尾句子，以目標字元數（預設 250 字元，約 100-120 Tokens）為基準，
    由後向前累積完整句子，徹底避免抓取到純編號標籤或殘缺片語。
    """
    if not prev_chunk.strip():
        return ""

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[。！？\n])|(?<=[.!?])\s+", prev_chunk)
        if s.strip()
    ]
    if not sentences:
        return ""

    selected = []
    accumulated_len = 0
    for s in reversed(sentences):
        selected.insert(0, s)
        accumulated_len += len(s)
        if accumulated_len >= target_overlap_chars:
            break

    prefix = " ".join(selected)
    if not prefix.endswith(("。", "！", "？", ".", "!", "?")):
        prefix += "。"
    return prefix


def chunk_text_with_overlap(
    text: str,
    target_overlap_chars: int = 250,
    max_safe_length: int = 1000,
    chunker_type: str = "semantic",
    initial_prev_chunk: str = ""
) -> list[str]:
    """
    純文字切分處理：
    - semantic: 先執行語意切分（SemanticChunker），對超過 max_safe_length 之區塊執行遞迴字元兜底切分。
    - recursive: 直接使用 RecursiveCharacterTextSplitter 進行高速字元切分。
    並自動為後續區塊補充前一區塊累積約 100-120 Tokens（250 字元）之完整句子以形成上下文重疊。
    支援跨切塊 / 跨檔案前綴銜接（initial_prev_chunk）。
    """
    if not text.strip():
        return []

    if chunker_type == "semantic":
        raw_chunks = semantic_splitter.split_text(text)
        normalized_chunks = []
        for chunk in raw_chunks:
            if len(chunk) > max_safe_length:
                sub_splits = safety_fallback_splitter.split_text(chunk)
                normalized_chunks.extend(sub_splits)
            else:
                normalized_chunks.append(chunk)
    else:
        normalized_chunks = safety_fallback_splitter.split_text(text)

    if not normalized_chunks:
        return []

    final_chunks = []
    # 若有傳入跨切塊/跨檔案前置文字，為第一個切塊加上重疊前綴
    if initial_prev_chunk:
        overlap_prefix = extract_overlap_prefix(initial_prev_chunk, target_overlap_chars)
        if overlap_prefix:
            final_chunks.append(f"{overlap_prefix}\n{normalized_chunks[0]}")
        else:
            final_chunks.append(normalized_chunks[0])
    else:
        final_chunks.append(normalized_chunks[0])

    for i in range(1, len(normalized_chunks)):
        prev_chunk = normalized_chunks[i - 1]
        curr_chunk = normalized_chunks[i]

        overlap_prefix = extract_overlap_prefix(prev_chunk, target_overlap_chars)
        if overlap_prefix:
            final_chunks.append(f"{overlap_prefix}\n{curr_chunk}")
        else:
            final_chunks.append(curr_chunk)

    return final_chunks


def init_database(conn: psycopg.Connection):
    """確保資料表與 pgvector 擴充套件存在，並建立 HNSW 向量索引與輔助索引"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manual_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_id VARCHAR(120) UNIQUE NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                chunk_type VARCHAR(20) NOT NULL, -- 'text' 或 'table'
                chunk_index INT NOT NULL,
                chapter_path TEXT,
                content TEXT NOT NULL,
                embedding vector(1024) NOT NULL, -- 對應 bge-m3 輸出之 1024 維度
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            -- 補齊支援統一麵包屑路徑欄位
            ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS chapter_path TEXT;

            -- 建立全文檢索自動計算欄位（tsvector）
            -- 運作機制拆解：
            -- 1. to_tsvector('english', ...)：調用英文文法字典進行切詞、過濾贅詞（the, is）並還原詞根（mounting -> mount），但不設定權重。
            -- 2. setweight(..., 'A'/'B')：專責替切好的單字陣列蓋上重要性權重等級章（'A' 標記完整章節路徑 chapter_path，'B' 標記正文 content）。
            -- 3. 符號 ||：將標題路徑與內文兩份加權後的詞彙清單串接合併為單一 tsvector。
            -- 4. STORED 關鍵字：資料庫自動運算斷詞並持久化儲存，新增資料時自動維護，完全不需手動重新切塊。
            ALTER TABLE manual_chunks 
            ADD COLUMN IF NOT EXISTS tsv tsvector 
            GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(chapter_path, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(content, '')), 'B')
            ) STORED;

            -- 建立 HNSW 向量索引（供 AI 語意檢索計算餘弦相似度使用）
            CREATE INDEX IF NOT EXISTS idx_manual_chunks_hnsw 
            ON manual_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);

            -- 建立 GIN 倒排索引（供全文檢索與 BM25 關鍵字高速匹配使用）
            CREATE INDEX IF NOT EXISTS idx_manual_chunks_tsv ON manual_chunks USING gin(tsv);
            CREATE INDEX IF NOT EXISTS idx_manual_chunks_source ON manual_chunks (source_file);
            CREATE INDEX IF NOT EXISTS idx_manual_chunks_type ON manual_chunks (chunk_type);
        """)
    conn.commit()


def natural_sort_key(p: Path) -> list:
    """按自然數字順序排序檔名（例如 0, 1, 2... 10）"""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", p.name)
    ]


def clean_for_embedding(text: str) -> str:
    """
    清理用於計算向量之文字（移除 HTML 冗餘樣式屬性並進行長度防護），
    避免觸發 Ollama context limit 錯誤，資料庫中仍完整保留原始 content。
    """
    cleaned = re.sub(r"\s+style='[^']*'", "", text)
    cleaned = re.sub(r'\s+style="[^"]*"', "", cleaned)
    if len(cleaned) > 3500:
        cleaned = cleaned[:3500]
    return cleaned


def run_pipeline(
    output_dir_str: str = "./output",
    batch_size: int = 32,
    chunker_type: str = "semantic"
):
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute() and not output_dir.exists():
        fallback_dir = Path(__file__).resolve().parent.parent / output_dir_str.lstrip("./")
        if fallback_dir.exists():
            output_dir = fallback_dir

    if not output_dir.exists():
        raise FileNotFoundError(f"找不到輸入目錄：{output_dir.resolve()}")

    records_to_insert: list[dict] = []
    md_files = sorted(output_dir.glob("*.md"), key=natural_sort_key)
    print(f"發現 {len(md_files)} 份 Markdown 檔案，開始執行結構化切分流程（切分模式：{chunker_type}）...", flush=True)

    # 跨檔案上下文與章節延續器（Cross-File Context Carryover）
    active_header_stack: dict[int, str] = {}
    last_valid_chapter_path: str = "未分類章節"
    last_text_chunk_across_files: str = ""

    for file_idx, md_file in enumerate(md_files):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue

        # 階段一：提取標題與章節階層
        sections = header_splitter.split_text(content)
        chunk_idx = 0

        for sec in sections:
            sec_text = sec.page_content
            meta = sec.metadata

            # 跨檔案章節階層繼承機制
            current_headers = {}
            if meta:
                for k, v in meta.items():
                    m = re.match(r"^H(\d+)$", k)
                    if m and v and str(v).strip():
                        current_headers[int(m.group(1))] = str(v).strip()

            if current_headers:
                min_lvl = min(current_headers.keys())
                # 清除當前層級及以下的所有子標題，保留上位父層級
                active_header_stack = {lvl: title for lvl, title in active_header_stack.items() if lvl < min_lvl}
                active_header_stack.update(current_headers)
                chapter_path = build_breadcrumb_from_stack(active_header_stack)
                last_valid_chapter_path = chapter_path
            else:
                # 本區塊無標題（如跨頁承接段落），無縫繼承前頁有效章節
                chapter_path = last_valid_chapter_path

            # 階段二：分離 HTML 表格與非表格文字
            matches = list(HTML_TABLE_REGEX.finditer(sec_text))
            last_idx = 0

            for match in matches:
                start, end = match.span()
                pre_text = sec_text[last_idx:start].strip()

                # 階段三（A）：處理表格前的純文字
                if pre_text:
                    text_chunks = chunk_text_with_overlap(
                        pre_text,
                        target_overlap_chars=250,
                        chunker_type=chunker_type,
                        initial_prev_chunk=last_text_chunk_across_files
                    )
                    for text_chunk in text_chunks:
                        records_to_insert.append({
                            "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                            "source_file": md_file.name,
                            "chunk_type": "text",
                            "chunk_index": chunk_idx,
                            "chapter_path": chapter_path,
                            "content": text_chunk
                        })
                        chunk_idx += 1
                        last_text_chunk_across_files = text_chunk

                # 階段三（B）：處理表格本體（整張獨立入庫）
                table_html = match.group(1).strip()
                records_to_insert.append({
                    "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                    "source_file": md_file.name,
                    "chunk_type": "table",
                    "chunk_index": chunk_idx,
                    "chapter_path": chapter_path,
                    "content": table_html
                })
                chunk_idx += 1

                last_idx = end

            # 階段三（C）：處理剩餘之純文字
            tail_text = sec_text[last_idx:].strip()
            if tail_text:
                text_chunks = chunk_text_with_overlap(
                    tail_text,
                    target_overlap_chars=250,
                    chunker_type=chunker_type,
                    initial_prev_chunk=last_text_chunk_across_files
                )
                for text_chunk in text_chunks:
                    records_to_insert.append({
                        "chunk_id": f"{md_file.stem}_c{chunk_idx}",
                        "source_file": md_file.name,
                        "chunk_type": "text",
                        "chunk_index": chunk_idx,
                        "chapter_path": chapter_path,
                        "content": text_chunk
                    })
                    chunk_idx += 1
                    last_text_chunk_across_files = text_chunk

        if (file_idx + 1) % 10 == 0 or file_idx + 1 == len(md_files):
            print(f"[{file_idx + 1}/{len(md_files)}] 已完成切分預處理，累積生成 {len(records_to_insert)} 個區塊...", flush=True)

    total_chunks = len(records_to_insert)
    table_count = sum(1 for r in records_to_insert if r["chunk_type"] == "table")
    text_count = sum(1 for r in records_to_insert if r["chunk_type"] == "text")
    print(f"切分處理完成：共生成 {total_chunks} 個知識區塊（純文字區塊：{text_count}，表格區塊：{table_count}）。", flush=True)
    print(f"開始連線 PostgreSQL ({PG_CONN_STRING}) 並執行向量化入庫...", flush=True)

    # 階段四：連線資料庫並批次寫入
    with psycopg.connect(PG_CONN_STRING) as conn:
        # 先建立資料庫綱要與 vector 擴充，再註冊向量類型轉換器
        init_database(conn)
        register_vector(conn)

        for i in range(0, total_chunks, batch_size):
            batch = records_to_insert[i:i + batch_size]
            contents_for_embedding = [clean_for_embedding(item["content"]) for item in batch]

            # 批次計算 bge-m3 向量
            embeddings_vecs = langchain_embeddings.embed_documents(contents_for_embedding)

            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO manual_chunks (
                        chunk_id, source_file, chunk_type, chunk_index,
                        chapter_path, content, embedding
                    ) VALUES (
                        %(chunk_id)s, %(source_file)s, %(chunk_type)s, %(chunk_index)s,
                        %(chapter_path)s, %(content)s, %(embedding)s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chapter_path = EXCLUDED.chapter_path,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding;
                """, [
                    {**item, "embedding": vec}
                    for item, vec in zip(batch, embeddings_vecs)
                ])
            conn.commit()
            print(f"進度：已成功寫入 {min(i + batch_size, total_chunks)} / {total_chunks} 筆至 PostgreSQL", flush=True)

    print("全量資料寫入 PostgreSQL 完畢。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="技術手冊 RAG 結構化切分與 PostgreSQL (pgvector) 入庫管線")
    parser.add_argument("--output-dir", default="./output", help="輸入 Markdown 目錄路徑（預設：./output）")
    parser.add_argument("--batch-size", type=int, default=32, help="寫入資料庫之批次大小（預設：32）")
    parser.add_argument(
        "--chunker",
        choices=["semantic", "recursive"],
        default="semantic",
        help="切分策略模式：semantic (語意切分+兜底) 或 recursive (純遞迴字元切分)"
    )
    args = parser.parse_args()

    run_pipeline(
        output_dir_str=args.output_dir,
        batch_size=args.batch_size,
        chunker_type=args.chunker
    )
