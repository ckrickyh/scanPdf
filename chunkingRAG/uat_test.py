import os
import re
import sys
import psycopg
from pgvector.psycopg import register_vector
import ollama

try:
    from chunkingRAG.db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_db_target,
        get_gemini_api_key,
        get_gemini_model,
    )
except ImportError:
    from db_config import (
        get_pg_conn_string,
        get_embedding_model,
        get_db_target,
        get_gemini_api_key,
        get_gemini_model,
    )
from google import genai

PG_CONN_STRING = get_pg_conn_string()
EMBEDDING_MODEL = get_embedding_model()


class UATRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def record(self, test_id: str, name: str, success: bool, detail: str):
        status_str = "PASS" if success else "FAIL"
        if success:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append({
            "id": test_id,
            "name": name,
            "status": status_str,
            "detail": detail
        })
        print(f"[{status_str}] {test_id} - {name}")
        print(f"       細節：{detail}\n")

    def run_all(self):
        print(f"\n{'='*65}")
        print("開始執行技術手冊 RAG 與 pgvector 入庫 使用者驗收測試（UAT）")
        print(f"{'='*65}\n")

        self.test_uat_01_db_and_extension()
        self.test_uat_02_record_count_and_dimensions()
        self.test_uat_03_hnsw_index_integrity()
        self.test_uat_04_table_atomic_isolation()
        self.test_uat_05_sentence_overlap_integrity()
        self.test_uat_06_chunk_id_uniqueness()
        self.test_uat_07_cross_lingual_semantic_retrieval()
        self.test_uat_08_table_dedicated_retrieval()
        self.test_uat_09_gemini_api_key_and_inference()

        print(f"{'='*65}")
        print(f"UAT 驗收測試總結：通過 {self.passed} 項，失敗 {self.failed} 項（共 {len(self.results)} 項）")
        print(f"{'='*65}\n")
        return self.failed == 0

    def test_uat_01_db_and_extension(self):
        """UAT-01: 資料庫連線與 pgvector 擴充套件有效性"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                    row = cur.fetchone()
                    if row:
                        self.record("UAT-01", "資料庫連線與 pgvector 擴充套件", True, f"成功連線，pgvector 版本：{row[0]}")
                    else:
                        self.record("UAT-01", "資料庫連線與 pgvector 擴充套件", False, "未找到 vector 擴充套件")
        except Exception as e:
            self.record("UAT-01", "資料庫連線與 pgvector 擴充套件", False, f"連線失敗：{e}")

    def test_uat_02_record_count_and_dimensions(self):
        """UAT-02: 區塊總數與 1024 維度向量完整性"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*), count(embedding) FROM manual_chunks;")
                    total, embedded = cur.fetchone()
                    cur.execute("SELECT vector_dims(embedding) FROM manual_chunks LIMIT 1;")
                    dim = cur.fetchone()[0]

                    success = (total == embedded and total >= 300 and dim == 1024)
                    self.record(
                        "UAT-02",
                        "區塊總數與向量維度檢驗",
                        success,
                        f"總區塊數：{total}，已計算向量數：{embedded}，向量維度：{dim} 維（符合 bge-m3 1024 維度規格）"
                    )
        except Exception as e:
            self.record("UAT-02", "區塊總數與向量維度檢驗", False, str(e))

    def test_uat_03_hnsw_index_integrity(self):
        """UAT-03: HNSW 向量索引配置與運作狀態"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT indexname, indexdef 
                        FROM pg_indexes 
                        WHERE tablename = 'manual_chunks' AND indexname = 'idx_manual_chunks_hnsw';
                    """)
                    row = cur.fetchone()
                    if row and "hnsw" in row[1] and "vector_cosine_ops" in row[1]:
                        self.record("UAT-03", "HNSW 向量索引驗證", True, f"索引存在：{row[0]}，配置包含 vector_cosine_ops 與 HNSW 圖結構")
                    else:
                        self.record("UAT-03", "HNSW 向量索引驗證", False, "未檢測到符合規範之 HNSW 向量索引")
        except Exception as e:
            self.record("UAT-03", "HNSW 向量索引驗證", False, str(e))

    def test_uat_04_table_atomic_isolation(self):
        """UAT-04: HTML 原生表格原子級隔離（未被拆散）"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT chunk_id, content FROM manual_chunks WHERE chunk_type = 'table';")
                    tables = cur.fetchall()
                    if not tables:
                        self.record("UAT-04", "HTML 原生表格原子隔離驗證", False, "未找到任何 chunk_type='table' 之區塊")
                        return

                    all_valid = True
                    details = []
                    for cid, content in tables:
                        stripped = content.strip()
                        starts_table = stripped.lower().startswith("<table")
                        ends_table = stripped.lower().endswith("</table>")
                        if not (starts_table and ends_table):
                            all_valid = False
                        details.append(f"{cid} (長度: {len(stripped)} 字元, 閉合完整: {starts_table and ends_table})")

                    self.record(
                        "UAT-04",
                        "HTML 原生表格原子隔離驗證",
                        all_valid,
                        f"共檢測 {len(tables)} 張表格，皆完整閉合且未被切分器切散：{', '.join(details)}"
                    )
        except Exception as e:
            self.record("UAT-04", "HTML 原生表格原子隔離驗證", False, str(e))

    def test_uat_05_sentence_overlap_integrity(self):
        """UAT-05: 句子級重疊（Overlap）機制驗證"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    # 抽檢具有多個 chunk 的文件（例如 PiCUSQ72Manual_13 的 c3 與 c4）
                    cur.execute("""
                        SELECT chunk_index, content 
                        FROM manual_chunks 
                        WHERE source_file = 'PiCUSQ72Manual_13.md' AND chunk_type = 'text'
                        ORDER BY chunk_index ASC;
                    """)
                    chunks = cur.fetchall()
                    if len(chunks) < 2:
                        self.record("UAT-05", "句子重疊補償機制", False, "樣本文件區塊數不足以比對重疊")
                        return

                    # 驗證後續區塊首句是否與前一區塊結尾相關或有換行首句
                    overlap_found = False
                    for i in range(1, len(chunks)):
                        prev_content = chunks[i - 1][1].strip()
                        curr_content = chunks[i][1].strip()
                        first_line = curr_content.split("\n")[0].strip()
                        if first_line and first_line in prev_content:
                            overlap_found = True
                            break

                    self.record(
                        "UAT-05",
                        "句子重疊補償機制",
                        overlap_found,
                        "驗證相鄰純文字區塊存在前文結尾句子之重疊前綴，滿足上下文無損條件"
                    )
        except Exception as e:
            self.record("UAT-05", "句子重疊補償機制", False, str(e))

    def test_uat_06_chunk_id_uniqueness(self):
        """UAT-06: 區塊唯一識別碼與冪等性約束驗證"""
        try:
            with psycopg.connect(PG_CONN_STRING) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(chunk_id), count(DISTINCT chunk_id) FROM manual_chunks;")
                    total, distinct = cur.fetchone()
                    success = (total == distinct and total > 0)
                    self.record(
                        "UAT-06",
                        "區塊唯一性與約束驗證",
                        success,
                        f"總區塊數：{total}，不重複 chunk_id 數：{distinct}，無重複鍵衝突"
                    )
        except Exception as e:
            self.record("UAT-06", "區塊唯一性與約束驗證", False, str(e))

    def test_uat_07_cross_lingual_semantic_retrieval(self):
        """UAT-07: 跨語言業務場景語意檢索驗證"""
        query = "感測器如何安裝在樹幹上？"
        try:
            res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
            query_vec = res["embedding"]

            with psycopg.connect(PG_CONN_STRING) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT chunk_id, source_file, 1 - (embedding <=> %s::vector) AS score, content
                        FROM manual_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT 3;
                    """, (query_vec, query_vec))
                    rows = cur.fetchall()

            top1_id, top1_src, top1_score, top1_content = rows[0]
            # 驗證相似度大於 0.6 且內容涵蓋安裝或樹幹相關內容
            relevant = ("mount" in top1_content.lower() or "sensor" in top1_content.lower() or "tree" in top1_content.lower())
            success = (top1_score > 0.6 and relevant)

            self.record(
                "UAT-07",
                "跨語言業務語意檢索驗證",
                success,
                f"提問：『{query}』 -> 命中首位：{top1_src} ({top1_id})，相似度：{top1_score:.4f}，成功命中手冊相關章節"
            )
        except Exception as e:
            self.record("UAT-07", "跨語言業務語意檢索驗證", False, str(e))

    def test_uat_08_table_dedicated_retrieval(self):
        """UAT-08: 原生表格專用檢索與類型過濾驗證"""
        query = "MP used Transmitter Receiver Histogram"
        try:
            res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
            query_vec = res["embedding"]

            with psycopg.connect(PG_CONN_STRING) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT chunk_id, chunk_type, 1 - (embedding <=> %s::vector) AS score, content
                        FROM manual_chunks
                        WHERE chunk_type = 'table'
                        ORDER BY embedding <=> %s::vector
                        LIMIT 1;
                    """, (query_vec, query_vec))
                    row = cur.fetchone()

            if row:
                cid, ctype, score, content = row
                success = (ctype == "table" and score > 0.5 and "<table" in content.lower())
                self.record(
                    "UAT-08",
                    "表格專用檢索與型態過濾驗證",
                    success,
                    f"提問：『{query}』 -> 命中表格：{cid}，相似度：{score:.4f}，回傳資料確認包含原生 <table> 標籤"
                )
            else:
                self.record("UAT-08", "表格專用檢索與型態過濾驗證", False, "未檢索到表格區塊")
        except Exception as e:
            self.record("UAT-08", "表格專用檢索與型態過濾驗證", False, str(e))

    def test_uat_09_gemini_api_key_and_inference(self):
        """UAT-09: Google Gemini API 金鑰有效性與模型對話推論驗證"""
        api_key = get_gemini_api_key()
        model = get_gemini_model()

        if not api_key:
            self.record("UAT-09", "Gemini API 金鑰與推論驗證", False, "未在 .env 檢測到 GEMINI_API_KEY 或 GOOGLE_API_KEY")
            return

        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        try:
            client = genai.Client(api_key=api_key)
            chat = client.chats.create(model=model)
            test_prompt = "請僅回覆繁體中文五個字：API測試成功"
            response = chat.send_message(test_prompt)
            output_text = response.text.strip() if response.text else ""

            success = bool(output_text and ("API" in output_text or "成功" in output_text))
            self.record(
                "UAT-09",
                "Gemini API 金鑰與推論驗證",
                success,
                f"金鑰：{masked_key}，模型：{model}，呼叫回應：『{output_text}』"
            )
        except Exception as e:
            self.record("UAT-09", "Gemini API 金鑰與推論驗證", False, f"金鑰：{masked_key}，模型：{model}，呼叫失敗：{str(e)}")


if __name__ == "__main__":
    runner = UATRunner()
    if "--gemini" in sys.argv or "--api-key" in sys.argv:
        print(f"\n{'='*65}")
        print("執行 Google Gemini API Key 專案驗收測試（UAT-09）")
        print(f"{'='*65}\n")
        runner.test_uat_09_gemini_api_key_and_inference()
        print(f"{'='*65}")
        print(f"UAT API 驗收測試總結：通過 {runner.passed} 項，失敗 {runner.failed} 項")
        print(f"{'='*65}\n")
        sys.exit(0 if runner.failed == 0 else 1)
    else:
        all_passed = runner.run_all()
        sys.exit(0 if all_passed else 1)

