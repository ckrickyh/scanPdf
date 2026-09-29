# 技術手冊雙路混合檢索與 RAG 智慧問答系統（Technical Manual RAG）

本專案為針對儀器技術手冊（Technical Manual）設計之端到端檢索增強生成（Retrieval-Augmented Generation, RAG）系統，作為技術驗證與初步實作練習。系統針對 PDF 的文字與原生表格進行結構化抽取，結合雙路混合檢索（向量語意檢索與關鍵字全文檢索）、倒數排名融合（RRF）與交叉編碼器（Cross-Encoder）深度重排技術，並透過 Google Gemini 2.5 Flash 生成具備原文可追溯性之繁體中文技術解答。


---

## 一、 系統架構與核心特色

### 1. 視覺語言文件解析（Vision-Language OCR）
* 採用 PaddleOCR-VL-1.6 視覺語言模型，直接解析 PDF 掃描檔。
* 完整提取階層標題（H1 至 H5）並維持章節脈絡。
* 獨立抽離 HTML 原生表格（`<table>`），維持資料原子性，避免跨區塊切分導致數據錯位。

### 2. 雙層自適應切分機制（Two-Tier Chunking）
* **第一層（語意切分，SemanticChunker）**：以單句拆解文本，送入 Embedding 模型計算相鄰句之餘弦距離，當語意躍升超過第 85 百分位數門檻時進行切分，確保段落語意凝聚度。
* **第二層（字元安全兜底，Recursive Splitter）**：針對長篇平鋪直敘步驟，自動觸發二次防護切分（設定 900 字元上限、150 字元重疊），嚴格防範向量稀釋。
* **上下文回溯補償**：向前回溯前一區塊結尾之完整句子作為重疊前綴（約 200 字元），確保切塊間語意銜接完整。

### 3. 雙路混合檢索與 RRF 排名融合（Hybrid Search & RRF）
* **向量檢索通道（Dense Retrieval）**：PostgreSQL 配合 pgvector 套件，建立 1024 維度 HNSW 索引（使用餘弦距離度量），並設置最低相似度門檻（`min_similarity >= 0.5`）阻斷不相關雜訊。
* **全文檢索通道（Sparse Retrieval）**：基於 PostgreSQL GIN 倒排索引與 BM25 演算法，針對標題（權重等級 A，1.0）與內文（權重等級 B，0.4）進行加權計算，鎖定型號代碼與特殊術語。
* **RRF 倒數排名融合**：結合兩路名次進行無量綱融合評分，公式為 $Score = \frac{1}{60 + Rank_{\text{vec}}} + \frac{1}{60 + Rank_{\text{kw}}}$，自全量資料中海選出 15 至 30 筆初篩候選切塊（Candidate Pool）。

### 4. 交叉編碼器重排（Cross-Encoder Reranker）
* 整合 `BAAI/bge-reranker-v2-m3` 交叉編碼模型，將「使用者問題」與「候選區塊」逐對組裝為 `[CLS] 問題 [SEP] 內文 [SEP]` 序列。
* 透過全注意力機制（Cross-Attention）進行字對字關聯比對，經 Sigmoid 正規化計算關聯分數，精選出 Top-K 核心切塊，大幅降低大型語言模型幻覺。

### 5. 雙向資料庫切換與雙軌推論支援
* 支援切換本地 PostgreSQL 或線上 Supabase pgvector 資料庫。
* 向量計算支援本機 Ollama 服務與 Hugging Face Sentence Transformers 本地推論自動降級機制。

### 6. LLM-as-a-Judge 自動化問答評審機制（Quality Evaluation）
* 整合以 Google Gemini 為評判者之自動化品質評估模組（`evaluation/llm_judge.py`）。
* 採用階梯式評分規約（Rubrics）與 Pydantic 結構化約束，針對「脈絡相關性」、「忠實度（幻覺檢測）」、「答案相關性」與「完整度」四維度進行強制思維鏈推導與量化評分。
* 內建抗網路重置（Connection Reset）、503 服務超載與多模型自動降級備援機制，確保長文本評審之穩定性。

### 7. 外部知識洩漏稽核器（External Knowledge Leakage Detector）
* 獨立安全稽核模組（`evaluation/leakage_detector.py`），專門檢驗問答是否受限於檢索到的 PDF 切塊（Context Isolation）。
* 採用原子事實分解（Atomic Claim Decomposition）與嚴苛自然語言推理（NLI）蘊含比對，量化計算外部知識洩漏率（Leakage Rate），杜絕模型偷偷使用預訓練先驗記憶回答非手冊內容。

---

## 二、 系統架構流程圖

```
【離線建庫階段】
[PDF 手冊] ──▶ [PaddleOCR-VL] ──▶ [Markdown / HTML Table 結構化]
                                        │
                                        ▼
                            [雙層切分架構與上下文補償]
                                        │
                                        ▼
                             [BGE-M3 向量化 (1024 維)]
                                        │
                                        ▼
                       [PostgreSQL / Supabase (pgvector)]
                       ├── 建立 HNSW 向量索引 (Cosine)
                       └── 建立 GIN 倒排全文索引 (tsvector)

【線上檢索與問答階段】
[使用者提問] ──▶ [BGE-M3 提問向量化]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[HNSW 向量語意檢索]          [GIN 關鍵字全文檢索]
(餘弦相似度過濾 >= 0.5)      (BM25 詞頻加權: 標題 A / 內文 B)
         │                           │
         └─────────────┬─────────────┘
                       ▼
            [RRF 倒數排名融合初篩]
            (取 Top 15~30 候選池)
                       │
                       ▼
          [Cross-Encoder 深度重排]
          (bge-reranker-v2-m3 逐對交叉注意力)
                       │
                       ▼
            [精選 Top-K 切塊組裝 Prompt]
                       │
                       ▼
          [Google Gemini 2.5 Flash]
          (設定 Temperature=0.2, Top-P=0.85)
                       │
                       ▼
             [高精確度繁體中文技術解答]
```

---

## 三、 環境需求與安裝前置作業

### 1. 執行環境需求
* 作業系統：macOS、Linux 或 Windows。
* Python 版本：Python 3.12（強制遵循 uv 規範）。
* 套件管理工具：[uv](https://docs.astral.sh/uv/)。
* 關聯式資料庫：PostgreSQL 15+（需啟用 pgvector 擴充功能），或線上 Supabase 雲端實例。
* 本地向量推論服務（選用）：Ollama（需預先下載 `bge-m3` 模型）。

### 2. 專案依賴安裝
使用 `uv` 進行虛擬環境建立與依賴同步：

```bash
# 鎖定 Python 3.12 基準環境並同步依賴套件
uv python pin 3.12
uv sync
```

### 3. 環境變數配置
專案根目錄建立 `.env` 檔案，可自 `.env.example` 複製後填入配置參數：

```bash
cp .env.example .env
```

`.env` 參數定義說明：

| 變數名稱 | 必填 | 說明 | 範例值 |
| :--- | :--- | :--- | :--- |
| `DB_TARGET` | 是 | 資料庫連線目標，支援 `local` 或 `supabase` | `supabase` |
| `LOCAL_PG_CONN_STRING` | 否 | 本機 PostgreSQL 連線字串 | `postgresql://postgres:postgres@localhost:5432/scan_pdf_rag` |
| `SUPABASE_PG_CONN_STRING` | 是 | 線上 Supabase PostgreSQL 連線字串（需使用 Session Pooler 5432 埠位並包含 `?sslmode=require`） | `postgresql://postgres.[REF]:[PWD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require` |
| `EMBEDDING_MODEL` | 是 | 向量嵌入模型名稱 | `bge-m3` |
| `OLLAMA_BASE_URL` | 否 | Ollama API 服務位址 | `http://localhost:11434` |
| `GEMINI_API_KEY` | 是 | Google Gemini API 金鑰 | `AIzaSy...` |
| `GEMINI_MODEL` | 是 | 呼叫之大型語言模型代碼 | `gemini-2.5-flash` |

### 4. 資料庫初始化
若使用全新資料庫或 Supabase 專案，需先於資料庫執行結構定義腳本。使用 Supabase 時，請至 SQL Editor 貼上並執行 `supabase_init.sql` 內容；使用本機 PostgreSQL 時，可直接透過 `psql` 執行：

```bash
psql -U postgres -d scan_pdf_rag -f supabase_init.sql
```

該腳本將建立：
1. `vector` 擴充套件。
2. `manual_chunks` 切塊儲存資料表。
3. `idx_manual_chunks_hnsw` 向量索引（1024 維餘弦度量）。
4. `idx_manual_chunks_tsv` 倒排索引（`tsvector` 自動計算欄位，針對 H1 至 H5 與內文進行加權分級）。

---

## 四、 使用說明（Usage Guide）

### 步驟 1：掃描 PDF 手冊並抽取結構化資料
將欲處理之 PDF 技術手冊放入 `scanPdf/source/` 目錄，執行視覺語言模型解析腳本：

```bash
uv run python scanPdf/scan_pdf.py
```
* 輸出成果將儲存於 `output/` 目錄下，包含結構化 Markdown（`.md`）與 JSON（`.json`）格式檔案。

### 步驟 2：執行切分、向量化並匯入資料庫
執行資料處理管線，自動進行 Markdown 階層標題切分、原生 HTML 表格抽取、雙層語意切分、上下文重疊補償與向量入庫：

```bash
# 匯入至當前 .env 指定之目標資料庫（local 或 supabase）
uv run python chunkingRAG/pipeline_postgres.py --input-dir output
```

### 步驟 3：啟動 Gradio 互動檢索問答平台
啟動圖形化問答展示介面：

```bash
# 1. 本地內網展示模式（預設位址 http://localhost:7860）
uv run python app.py

# 2. 臨時外網分享模式（自動建立 72 小時公開存取連結）
GRADIO_SHARE=true uv run python app.py
```

介面核心控制選項說明：
* **區塊類型篩選**：可指定檢索全部、純文字（`text`）或原生表格（`table`）。
* **初篩候選數（Candidate Pool）**：控制第一階段由向量與全文檢索撈出之候選數量（建議設定 15 至 30 筆）。
* **啟用混合檢索（Hybrid Search）**：同時啟用向量語意與 BM25 關鍵字檢索，並經由 RRF 融合。
* **啟用 Reranker**：啟用 `bge-reranker-v2-m3` 交叉注意力精排模型。
* **最終精選筆數（Top-K）**：傳遞至大型語言模型之參考切塊數量（預設 3 筆）。
* **回覆篇幅偏好**：可切換「精簡條列」、「標準詳細」或「深入完整」，精準控制生成 Token 上限與論述深度。

### 步驟 4：執行驗證與品質測試（選用）
系統提供專屬評測套件目錄 `evaluation/`，用於評估檢索召回成效、系統功能健全性、端到端生成品質與外部知識洩漏防護：

```bash
# 1. 執行檢索召回評估（比對純向量、純關鍵字、混合檢索與 Reranker 之 Recall/Precision/MRR）
uv run python evaluation/recall_check.py

# 2. 執行使用者驗收測試（UAT 測試集自動化評測）
uv run python evaluation/uat_test.py

# 3. 執行 LLM-as-a-Judge 品質評審（四維度：脈絡相關性、忠實度、答案相關性、完整度）
uv run python evaluation/llm_judge.py --query "感測器如何安裝在樹幹上？" --top-k 3

# 4. 執行外部知識洩漏稽核（原子事實拆解與檢索切塊上下文隔離檢驗）
uv run python evaluation/leakage_detector.py --query "感測器如何安裝在樹幹上？" --top-k 3
```

評估報告包含判定結果（PASS / FAIL）、指標量化評分與思維鏈推導扣分理由。

---

## 五、 線上展示部署（Hugging Face Spaces 部署指南）

本專案提供自動化部署方案，架構設計遵循輕量化隔離原則，將線上展示端與離線重型套件完全分離。

### 1. 架構隔離策略
* **重型離線套件隔離**：本機環境包含 PaddleOCR、PaddlePaddle 與 LangChain 切分器（容量超過 3 GB）；部署腳本自動生成精簡鏡像目錄 `.deploy_space/`，線上端僅安裝 `gradio`、`sentence-transformers`、`torch` 與資料庫用戶端，容量壓縮至 500 MB 以內，建置時間由 10 分鐘縮短至 2 分鐘。
* **連線池協定決策**：線上端串接 Supabase 時，必須採用 **Session Pooler（5432 埠位）**，避免 PgBouncer Transaction Pooler 在每次交易結束時清除自訂向量型別解碼器註冊狀態。

### 2. Hugging Face Space 前置設定
1. 於 [Hugging Face Spaces](https://huggingface.co/spaces) 建立新 Space，SDK 類型選擇 **Gradio**。
2. 進入 Space 的「Settings」分頁，於「Variables and secrets」區塊加入以下環境變數與密鑰：
   * `DB_TARGET` = `supabase`
   * `SUPABASE_PG_CONN_STRING` = `postgresql://postgres.[REF]:[PWD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require`
   * `GEMINI_API_KEY` = `[您的 Google Gemini API 金鑰]`
   * `GEMINI_MODEL` = `gemini-2.5-flash`
   * `EMBEDDING_MODEL` = `bge-m3`

### 3. 一鍵自動化打包與部署指令
本專案提供專用部署工具 `scripts/deploy_space.py`，負責環境安全檢查、輕量鏡像生成與遠端推送作業。

```bash
# 步驟 A：乾跑檢查（僅於本機建置 .deploy_space/ 目錄，不執行遠端推送）
uv run python scripts/deploy_space.py --dry-run

# 步驟 B：設定 Hugging Face 遠端端點（首次部署需設定）
git remote add space https://huggingface.co/spaces/[您的使用者名稱]/[您的Space名稱]

# 步驟 C：執行一鍵同步推送
uv run python scripts/deploy_space.py --remote space --branch main
```

部署腳本自動執行之防護流程：
1. **安全預檢**：確認 `.env` 檔案未被 Git 追蹤，杜絕金鑰洩漏風險。
2. **目錄組裝**：建立 `.deploy_space/` 目錄，僅打包 `app.py`、`chunkingRAG/db_config.py`、`chunkingRAG/reranker.py`、專用 `requirements.txt` 與含有 YAML 中繼資料之 `README.md`。
3. **獨立推送**：於 Staging 目錄中獨立執行 Git 推送，不污染母專案版本庫歷程。

