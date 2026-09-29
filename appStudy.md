# Logic

## ScanPdf (PDF 掃描與入庫前處理階段)
- Vision Language Model (VLM) 掃描 PDF，抽取文本與 HTML 表格資訊
- 自動修復 OCR 辨識瑕疵與文字對齊

## Embedding（資料入庫階段）
- 結構預處理：
  - Markdown 標題切分：依據 H1、H2、H3、H4、H5 階層保留章節元數據
  - 原生 HTML <table> 獨立抽取為完整表格區塊，維持原子性不被切斷
- 雙層切分架構（Semantic Chunking + Safety Fallback）：
  - **第一層：語意切分（SemanticChunker）**：
    - 將長文本按單句拆解，送入 Embedding 模型（bge-m3）計算相鄰句子之間的向量餘弦距離（Cosine Distance）。
    - 依據設定之百分位數門檻（`breakpoint_threshold_type="percentile"`, `breakpoint_threshold_amount=85`），當相鄰句子的語意距離躍升超過第 85 百分位數時，判定為話題轉折點並精確下刀。
  - **第二層：字元安全兜底（Recursive Fallback）**：
    - 若遇長篇平鋪直敘之操作步驟，語意無顯著跳躍，單一語意區塊可能過長（> 1000 字元）。
    - 系統自動觸發 `RecursiveCharacterTextSplitter`（`chunk_size=900`, `chunk_overlap=150`）進行二次防護切分，嚴格防範超出 Transformer（512/8192 Tokens）輸入限制與向量稀釋。
  - **切分策略對比（Semantic vs Recursive）**：
    | 切分策略 | 運作原理 | 優點 | 限制 / 代價 | 適用場景 |
    | :--- | :--- | :--- | :--- | :--- |
    | **SemanticChunker** | 計算相鄰單句之向量餘弦距離，遇語意躍升處下刀 | 段落語意凝聚度極高，不切斷完整主題 | 入庫前需調用數百次 Embedding 計算距離，耗時增加 5～10 倍 | 技術說明、理論手冊、問答知識庫 |
    | **RecursiveTextSplitter** | 依階層標點（`\n\n`, `\n`, `。`, 空格）按固定字數切分 | 切分速度極快（毫秒級），區塊長度高度均勻 | 易在主題未完處強行切斷，造成跨區塊語意破碎 | 巨量日誌、代碼庫、快速原型驗證 |
- 上下文重疊補償：
  - 以正規表達式（re）向前回溯前一個 Chunk 結尾之完整句子（目標約 250 字元 / 100-120 Tokens）銜接至次區塊，確保語意連貫
  - 若該句超過 200 Tokens，則向前保留約 200 Tokens 作為重疊前綴（使重疊內容佔比小於 15%，防止主句語意被稀釋，同時提升檢索連貫性）
- 向量計算（Embedding）：
  - 將已完成上下文拼接的「完整文字區塊」送入 Attention 網路（bge-m3 模型），輸出 1024 維稠密向量（Dense Vector）
- 資料庫存入（DB Ingestion）：
  - **初始化時序防錯**：連線時必須先執行 `init_database(conn)` 確保 `CREATE EXTENSION IF NOT EXISTS vector;` 生效，其後方可調用 `register_vector(conn)`，防止新資料庫環境拋出 `vector type not found` 異常。
  - 將文本內容與向量寫入 PostgreSQL 的 manual_chunks 資料表 （postgre db supabase_iti.sql的架構）
  - *建立 HNSW* 向量索引表：以餘弦距離（Cosine Distance）為度量尺規，將語意相近的句子節點互相連結，建立多層導航圖形網絡, 此索引 對應 postgre column 'embedding'。
  - json視覺化為例：
    ```
    {
      "entry_point": "行號 #1",
      "layers": {
        "layer_1": {
          "行號 #1": { "neighbors": ["行號 #3"] },
          "行號 #3": { "neighbors": ["行號 #1"] }
        },
        "layer_0": {
          "行號 #1": { "vector_id": 1, "neighbors": ["行號 #2"] },
          "行號 #2": { "vector_id": 2, "neighbors": ["行號 #1"] },
          "行號 #3": { "vector_id": 3, "neighbors": ["行號 #4"] },
          "行號 #4": { "vector_id": 4, "neighbors": ["行號 #3"] }
        }
      }
    }
    ```
  - *建立 GIN* 倒排索引表（Generalized Inverted Index）：於初始化資料庫時建立全文索引結構（`to_tsvector` 搭配 `USING gin(tsv)`）；每當切塊寫入時，資料庫會自動將文字拆解成單字字典並建立反查索引表，供後續線上檢索階段以 BM25 演算法進行毫秒級精準關鍵字比對。 此索引 對應 postgre column 'tsv'
    - 正常流程事例： 
      - 行號#1:「Ricky 的筆記本」裡面記了：香蕉、蘋果、牛奶。
      - 行號#2:「Alex 的筆記本」裡面記了：蘋果、西瓜。
    - 倒排索引（GIN 建立的反查表）：
      - 香蕉 $\rightarrow$ 出現在：Ricky ──▶ 指向 [行號 #1] 
      - 蘋果 $\rightarrow$ 出現在：Ricky、Alex ──▶ 指向 [行號 #1，2] 
      - 牛奶 $\rightarrow$ 出現在：Ricky ──▶ 指向 [行號 #1] 
      - 西瓜 $\rightarrow$ 出現在：Alex ──▶ 指向 [行號 #2] 
    - Gin優點：一搜「蘋果」，直接 0 秒查出 Ricky 和 Alex，完全不用翻筆記。
      

## LLM & Retrieval（檢索與問答階段）
- 問題向量化：提問問題經 bge-m3 模型轉換為 1024 維查詢向量
- 兩階段檢索架構（Two-Stage Retrieval with Reranker）：
  - 第一階段粗篩（High Recall）：由向量檢索自資料庫撈出 15 至 20 筆候選區塊（Candidate Pool）。 雙通道評審機制 （整體句子比較）：
    - 評審 A（HNSW 向量通道 = Cosine 相似度）：負責看「語意與氛圍」。就算你問「感測器怎麼裝」，文件寫「儀器固定步驟」，兩者方向一致（夾角小，Cosine 相似度高），就能被抓出來。
      - 相似度過濾門檻
        - 加入 min_similarity >= 0.5 條件，驗證無關提問（如義大利披薩）精確回傳 0 筆，阻斷雜訊干擾。
        ```sql
        WITH vector_search AS (
            SELECT 
                chunk_id,
                1 - (embedding <=> %(vec)s::vector) AS cosine_similarity
            FROM manual_chunks
            -- 評審 A 的專屬過濾條件：直接在向量通道擋掉低於 0.5 的資料
            WHERE 1 - (embedding <=> %(vec)s::vector) >= 0.5
            LIMIT 20
        )
        ```
    - 評審 B（GIN 關鍵字通道 = BM25 關鍵字比對）：負責看「精準硬指標」。專門盯著型號代碼（如 PiCUS Q72），確保專有名詞一個字都不漏。
      - 檢視關鍵字在postgres每一行出現的頻率
      - 字詞罕見度（字在所有文件出現的次數，越罕見越重要）
      - 標題加權（Weighting）：專案設定標題權重為 A（加乘 1.0），內文為 B（加乘 0.4），出現在章節標題的文章優先排在前面。
      ```sql
      WITH keyword_search AS (
          SELECT 
              chunk_id,
              content,
              -- 1.【BM25 計分】：計算關鍵字在文章中的相關度權重分數
              ts_rank_cd(tsv, plainto_tsquery('english', 'PiCUS Q72')) AS bm25_score,
              -- 2.【排序】：依照 BM25 分數由高至低排出名次（Rank）
              ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', 'PiCUS Q72')) DESC) AS rank_kw
          FROM manual_chunks
          -- 3.【GIN 索引過濾】：只挑出包含該關鍵字的切塊
          WHERE tsv @@ plainto_tsquery('english', 'PiCUS Q72')
          LIMIT 20
      )
      ```
    - RRF 計分板（Reciprocal Rank Fusion）：綜合評審 A 與 B 的排名，計算 $Score = \frac{1}{60 + Rank_{A}} + \frac{1}{60 + Rank_{B}}$，挑出前 15 至 20 筆候選。
      ```sql
      SELECT 
          COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
          COALESCE(v.content, k.content) AS content,
          -- RRF 融合打分公式
          (COALESCE(1.0 / (60 + v.rank_vec), 0.0) + COALESCE(1.0 / (60 + k.rank_kw), 0.0)) AS rrf_score
      FROM vector_search v
      FULL OUTER JOIN keyword_search k ON v.chunk_id = k.chunk_id
      ORDER BY rrf_score DESC
      LIMIT 20;
      ```
  - 第二階段精篩 Re-ranker（High Precision）：採用交叉編碼器（Cross-Encoder，`BAAI/bge-reranker-v2-m3`）進行一對一全注意力打分與重排：
    1. **輸入格式組裝（Pairing）**：將「使用者問題」與第一階段篩出的「15 筆候選切塊」分別拼接，組裝成 15 組輸入字串：
       `[CLS] 使用者問題 [SEP] 候選切塊內文 [SEP]`
       - **`[SEP]`（語意隔板）**：置於問題與內文之間及末尾，清楚劃分「前半段是題目」與「後半段是內文」，防止兩者語意混淆。
    2. **全注意力交叉質詢（Cross-Attention）**：直接把問題與文章關在同一個 Transformer 神經網路房間內，讓問題裡的每一個字與文章裡的每一個字當面互動理解（注意力矩陣相乘）。
    3. **提煉得分（Scoring with `[CLS]`）**：
       - **`[CLS]`（總結得分標記）**：置於最開頭，在運算過程中自動吸滿兩者交叉比對後的「整體匹配關聯精華（Interaction Summary）」。模型取其輸出經分類頭（Linear Head + Sigmoid）直接給出 $0.0 \sim 1.0$ 的精確相關度分數。
       ```text
       【Transformer 內部 6 × 1024 矩陣狀態與打分機制】
         ┌────────────────────────────────────────────────────────┐
         │ 第 0 列 [CLS] 向量: [0.34, -1.23, 0.05, ...共1024個數]  │ ──▶ 【只取這條去打分！】
         ├────────────────────────────────────────────────────────┤
         │ 第 1 列 "如"  向量: [0.12, 0.04, -0.91, ...共1024個數]  │ ──┐
         │ 第 2 列 "何"  向量: [-0.55, 0.61, 0.18, ...共1024個數] │   │
         │ 第 3 列 "安"  向量: [0.08, -0.29, 0.77, ...共1024個數] │   ├──▶ (打分時被丟棄，不使用)
         │ 第 4 列 [SEP] 向量: [0.91, -0.11, 0.44, ...共1024個數] │   │
         │ 第 5 列 "固"  向量: [-0.03, 0.81, 0.12, ...共1024個數] │   │
         │ 第 6 列 [SEP] 向量: [0.91, -0.11, 0.44, ...共1024個數] │ ──┘
         └────────────────────────────────────────────────────────┘
                                     │
                                     ▼ 只拿第 0 列算線性加權
                         Logit = W · [CLS] + b = 0.85
                                     │
                                     ▼ Sigmoid
                          相關度分數 = 0.94 (94%)
       ```
    4. **重新洗牌（Re-ranking）**：將 15 筆切塊依相關度分數由高至低重新排序，截取最高分的 Top-3 傳送給 LLM。
      - 套件就位：透過 uv add sentence-transformers 完成 PyTorch 與 Hugging Face 函式庫安裝。
      - 重排核心實作：建立 reranker.py 封裝 BAAI/bge-reranker-v2-m3，啟用 Apple Silicon MPS 晶片加速，實作 Sigmoid 正規化。
      - 檢索流程升級：資料庫粗篩 15 筆候選池（High Recall）-> Cross-Encoder 逐對計算交叉注意力並重排精選 Top-3（High Precision）。
- LLM答案生成（Deterministic Generation）：LLM（Google Gemini 2.5 Flash）綜合精篩後的 Top-K 參考文件，配置 `temperature=0.2` 與 `top_p=0.85`，壓制幻覺機率，產出高事實性繁體中文解答。


## Hybrid Search Architecture（雙路混合檢索架構）

| 順序與階段 | 步驟名稱（對應介面滑桿） | 篩選對象 | 執行角色 | 核心機制、演算法與【法官審案】比喻 |
| :--- | :--- | :--- | :--- | :--- |
| **【檢索階段】<br>第 1 步** | **雙路並行檢索** | 全手冊所有段落<br>（共 406 筆） | PostgreSQL 資料庫 | **書記官兵分兩路搜查法規**：<br>• `vector`（語意）：餘弦距離（`<=>`），按案情核心大意翻查相符條文。<br>• `tsvector`（字串）：BM25 詞頻（`ts_rank`），精準鎖定法規編號與專有名詞。 |
| **【檢索階段】<br>第 2 步** | **初篩海選**<br>（介面：初篩候選數 Candidate Pool） | **挑選文章（Chunks）**<br>（放寬至 15～30 筆） | 資料庫 RRF 融合演算法 | **書記官抱出候選卷宗（追求 High Recall）**：<br>• 透過 RRF 倒數排名融合公式：$\frac{1}{60 + \text{Rank}_{\text{vec}}} + \frac{1}{60 + \text{Rank}_{\text{kw}}}$。<br>• $\text{Rank}_{\text{vec}}$：文章在向量語意排行榜上的**名次**（如第 1 名代入 1）。<br>• $\text{Rank}_{\text{kw}}$：文章在關鍵字 BM25 排行榜上的**名次**（Keyword Rank，未上榜則該項為 0）。<br>• **為何看名次**：向量分數（餘弦值 0~1）與關鍵字分數（詞頻）單位不同，以名次倒數相加能公平融合。<br>• **寧可多抱、不可漏拿**，抱出 15 至 30 份候選案卷供法官助理審閱。 |
| **【檢索階段】<br>第 3 步** | **Reranker 重排決選**<br>（介面：最終精選筆數 Top-K） | **挑選文章（Chunks）**<br>（嚴格鎖定前 3～5 筆） | Cross-Encoder 模型<br>（`bge-reranker-v2-m3`） | **法官助理逐本審閱決選（追求 High Precision）**：<br>• 將案件與卷宗合併為 `[CLS]問題[SEP]內文[SEP]`，逐字交叉質詢注意力。<br>• 經 Sigmoid 打分，直接裁定選出最關鍵的 **前 3 本法條擺在法官桌上**！<br>• **【挑選書本／文章任務在此正式結束】**。 |
| **【生成階段】<br>第 4 步** | **LLM 答案生成**<br>（介面：Top-P 核採樣閥值） | **挑選單字（Tokens）**<br>（絕非挑選文章！） | Google Gemini 2.5 Flash | **主審法官手握鋼筆撰寫判決書（嚴謹用詞）**：<br>• 法官讀完桌上這 3 本法條，在判決書上逐字寫出回答。<br>• 經 Softmax 機率分配，`top_p=0.85` 規定法官落筆時只准從累積機率達 85% 的嚴謹法律詞彙中選字，嚴禁隨性胡說。 |

### tsvector 資料庫結構與語意解析
- **綱要定義與索引（pipeline_postgres.py: init_database）**：
  ```sql
  ALTER TABLE manual_chunks 
  ADD COLUMN IF NOT EXISTS tsv tsvector 
  GENERATED ALWAYS AS (
      setweight(to_tsvector('english', coalesce(h1, '') || ' ' || coalesce(h2, '') || ' ' || coalesce(h3, '') || ' ' || coalesce(h4, '') || ' ' || coalesce(h5, '')), 'A') ||
      setweight(to_tsvector('english', coalesce(content, '')), 'B')
  ) STORED;

  CREATE INDEX IF NOT EXISTS idx_manual_chunks_tsv ON manual_chunks USING gin(tsv);
  ```
- **儲存格式拆解（以 `'picus':1A,4B` 為例）**：
  * **格式規則**：`[單字字根]:[出現位置][權重等級]`
  * **數字（Word Position，單字出現位置）**：
    * `1` 代表是該段落的第 1 個單字，`4` 代表是第 4 個單字。
    * 用途：支援片語檢索（Phrase Search，如比對 `"picus sonic"` 時檢查位置 1 與 2 是否相鄰緊接）。
  * **字母（Weight Class，重要性權重等級）**：
    * `A`（最高權重）：指定給章節標題（`h1` 至 `h5`）。
    * `B`（次高權重）：指定給正文內文（`content`）。
    * 用途：計算 BM25 或 `ts_rank` 評分時，命中標題（A）的計分遠高於命中內文（B）。
- **更新機制**：使用 `GENERATED ALWAYS ... STORED` 儲存型生成欄位，資料庫引擎在建立欄位瞬間即自動補齊既有 406 筆資料，完全不需要重新切塊或重寫資料庫。

### Top-K vs Top-P（賽跑頒獎 vs 錄取及格）
- Top-K（看名次）：比賽「永遠只頒獎給前 3 名（K=3）」。不管第四名只差 0.01 秒，還是第一名贏第二名十條街，永遠只抓剛好 3 個人。
- Top-P（看實力佔比）：面試決定「錄取累積實力達到全體前 85%（P=0.85）的人」。若榜首一個人實力就獨佔 90%，這梯次就只錄取他 1 個人；若大家實力平分秋色，就可能一口氣錄取 8 個人。


## Evaluate RAG（檢索、生成與隔離安全評估體系）

系統將評估維度嚴格拆解為「檢索層」、「生成層」與「隔離安全層」，全方位監控 RAG 系統品質。

### 1. 檢索階段指標（Retrieval Stage）
由 `evaluation/recall_check.py` 執行自動化基準比對：
* **Recall@K**（$TP / (TP + FN)$）：衡量標準答案被成功抓回候選池的比例。RAG 系統優先追求 High Recall。
* **Precision@K**（$TP / (TP + FP)$）：衡量回傳切塊中真實相關資訊的純度。
* **Hit Rate@K**：前 K 筆結果中是否至少命中一筆標準參考答案。
* **MRR@K**（Mean Reciprocal Rank）：首個相關答案所在名次倒數之平均值，衡量高質量切塊是否排在最前列。

### 2. 生成階段評估：LLM-as-a-Judge（Generation Stage）
由 `evaluation/llm_judge.py` 執行，以高階模型（Google Gemini）作為獨立審查員，並採用階梯式評分規約（Rubrics）與思維鏈（CoT）：
* **脈絡相關性（Context Relevance）**：檢驗檢索切塊與使用者問題的聚焦程度，抓取冗餘雜訊。
* **忠實度（Faithfulness / 幻覺檢測）**：檢驗生成答案是否百分之百來自檢索資料，嚴格扣除未提及之外部猜測與幻覺。
* **答案相關性（Answer Relevance）**：檢驗回答是否正面切中使用者提問核心，排除答非所問。
* **完整度（Completeness）**：檢驗切塊內提供的必要操作步驟或數值規格是否被完整交代。
* **判定機制**：強制要求「先在 reasoning 寫出扣分依據與對照文字，最後才給出 score」；四指標皆達 4 分以上判定為 PASS，否則為 FAIL。

### 3. 安全層隔離評估：外部知識洩漏稽核器（External Knowledge Leakage Detector）
由 `evaluation/leakage_detector.py` 執行，防範模型動用預訓練記憶中的非手冊常識回答：
* **封閉世界假設（Closed-World Assumption）**：假設除了檢索到的參考切塊外，外部世界的一切事實皆不存在。
* **原子事實分解（Atomic Claim Decomposition）**：將受測答案拆解為單一獨立陳述句（Statements）。
* **上下文蘊含驗證（Context Entailment Check）**：逐句比對切塊文字，若在切塊中無明確記載（即使在現實世界為正確常理），一律判定為「外部先驗知識洩漏」。
* **外部洩漏率（Leakage Rate）**：計算未獲切塊支撐的陳述佔比，要求必須達到 0.0% 洩漏始為合格（PASS_ISOLATED）。

### 4. RRF 混合檢索足跡解析（以 `(向量: #5, 關鍵字: 未入榜)` 為例）
當日誌顯示：`【第 1 名】向量相似度：0.4931 | RRF 得分：0.01538 (向量: #5, 關鍵字: 未入榜) | Rerank 排序分：0.5079`
* **向量: #5**：該段落在 1024 維語意距離計算中名列第 5 名。
* **關鍵字: 未入榜**：因使用者提問詞彙未直接出現在該段落，BM25 未能排入前列候選。
* **RRF 數學推導**：$RRF = \frac{1}{60 + 5} + 0 = \frac{1}{65} \approx 0.01538$。
* **Reranker 逆轉**：雖然關鍵字未命中，但經由 Cross-Encoder 深度比對，注意力權重高達 0.5079，成功逆轉晉升為總評第 1 名。

### 5. 推論雙軌容錯機制
`chunkingRAG/search_manual.py` 與 `app.py` 具備自適應向量推論能力：
* 優先透過 `ollama.embeddings` 連線本地推論服務。
* 若連線中斷（如本機未啟動 Ollama 服務），自動降級切換至 `SentenceTransformer("BAAI/bge-m3")` 進行本地計算，避免中斷檢索流程。

## Security & Dependency Audit（套件依賴與資安審計）
- **核心目標**：確保專案所引入的第三方開源依賴（如 PyTorch、Gradio、psycopg 等）無已知 CVE 漏洞與後門風險。
- **無污染即時調用（uvx 特性）**：
  - 使用 `uvx` 於獨立的臨時沙盒環境中執行審計工具，**完全不污染專案本體的 `pyproject.toml` 與依賴鎖定檔（`uv.lock`）**。
- **審計執行指令**：
  ```bash
  # 掃描當前環境中所有依賴套件之已知漏洞 (CVE / OSV 資料庫比對)
  uvx pip-audit
  ```



# Supabase 連線模式: Session Pool 5432
- 連線模式為長連線：Gradio 或 FastAPI 後端啟動時，通常保持一個穩定的連線池，並發量通常在幾十到幾百個請求內，完全不會吃滿 Supabase 的 Session 配額。
  * 優點：極快（支援 Prepared Statements 快取，無額外解析開銷）
  * Trade-off：店裡只有 50 張桌子（連線數上限）。如果 50 個人坐著慢慢喝水聊天（連線沒關閉），第 51 位客人就必須在門口排隊，容量受限。
- 不是 Transaction Pool 6543
  * 優點：同一個櫃檯 1 小時可以服務 1000 位客人（超高並發）
  * Trade-off：只能點固定套餐，不能要求客製化座位服務（不支援複雜 Session 狀態與預編譯語句）。
  * 用於 Serverless 函式（如 AWS Lambda、Vercel Edge）

# Session pool 注意事項
- 盲點：使用 Session 模式時，若在 Python 寫了 conn = psycopg.connect(...) 卻沒有使用 with 語法或手動 conn.close()，該連線會一直佔據 Supabase 的名額。
- 防範解法：永遠使用 Python 的 Context Manager 管理連線生命週期

```
with psycopg.connect(CONN_STRING) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT ...")
# 離開縮排區塊自動關閉並釋放連線
```

# data 從本地 ocr 後的 md檔資料，完整搬移到 Supabase

1. 在 Supabase 初始化資料表結構

* 前往 Supabase 控制台的 SQL Editor。
* 開啟專案根目錄的 `supabase_init.sql`，複製全文並貼上執行。
* 此步驟會啟用 vector 擴充套件，並建立 manual_chunks 資料表與 HNSW / GIN 索引。

2. 確認 .env設定
* 確認將 DB_TARGET 設為 supabase。
* 確認連線字串包含真實密碼、連接埠 5432 與 ?sslmode=require。

3. 執行向量入庫管線
* 確保本地 Ollama 服務已啟動並下載 bge-m3 模型。
* 透過 uv 執行入庫腳本，將 output/ 內的 Markdown 文件切塊並寫入 Supabase。


## Test
uv run python chunkingRAG/search_manual.py --query "How to mount the sensor on tree?" --rerank

## RAG LLM-as-a-Judge
uv run evaluation/llm_judge.py \
  --candidate-model gemini-2.5-flash \
  --judge-model gemini-1.5-pro \
  --query "感測器如何安裝在樹幹上？" \
  --top-k 3

## RAG External Knowledge Leakage Audit
uv run evaluation/leakage_detector.py --query "感測器如何安裝在樹幹上？" --top-k 3

## NoSQL（MongoDB）雙資料庫架構與選型邏輯

### 1. 一句話白話定位
* **PostgreSQL (Supabase)**：負責「**大腦核心檢索**」（手冊向量、文字切塊、關鍵字混合檢索）。要求嚴謹格式與極致計算精度。
* **MongoDB (NoSQL)**：負責「**對話歷程與系統日誌**」（使用者問答會話、各階段檢索延遲日誌）。要求彈性結構、高頻追加寫入與自動過期清理。

### 2. 生活通俗比喻
* **PostgreSQL** 就像「**正式合約與精密病歷庫**」：每一頁格式都釘死，不可隨意塗改，適合存絕對不能出錯的手冊知識。
* **MongoDB** 就像「**隨身活頁便利貼本**」：
  * 每多聊一輪對話，就像往本子後面多貼一張便利貼（原生 `$push` 追加），不用整本重抄。
  * 今天想多記一個欄位（例如：模型延遲、使用者評分），直接寫上去即可，不需要重新印刷整本書（免改 Schema）。
  * 30 天過期的日誌，時間到了本子會自己化解消失（原生 TTL 索引自動清理），不需要人工手動撕紙。

### 3. 為什麼不直接用 PostgreSQL 的 JSONB 存日誌？
| 比較維度 | PostgreSQL (JSONB) | MongoDB (NoSQL) | 為什麼此場景選 MongoDB |
| :--- | :--- | :--- | :--- |
| **對話更新機制** | 每次追加對話，底層必須整行複製重寫（MVCC 寫入放大與表格膨脹）。 | 支援文件內原地更新（In-place `$push`），直接掛載到陣列尾端。 | 對話輪次頻繁更新時，MongoDB 磁碟 I/O 成本極低。 |
| **快取資源保護** | 日誌寫入與向量檢索搶奪同一塊記憶體緩衝區（Buffer Pool）。 | 獨立實體或獨立引擎，日誌寫入絕不干擾手冊檢索的 HNSW 向量快取。 | 避免日誌流量衝擊核心手冊檢索的反應速度（SLA）。 |
| **過期資料清理** | 需透過外部腳本定時執行大量 `DELETE`，容易造成長鎖與效能抖動。 | 內建 TTL 索引（Time To Live），底層非同步自動回收空間。 | 零運維成本，一行設定即實現 30 天日誌自動淘汰。 |
| **巢狀資料分析** | 分析多輪對話需使用多層 `LATERAL JOIN` 與子查詢，耗費 CPU。 | 原生 Aggregation Pipeline（`$unwind` 搭配 `$group`）串流運算。 | 輕鬆產出 P95 延遲、使用者滿意度與高頻搜尋詞報表。 |

### 4. 存檔結構視覺化範例

* **會話資料（chat_sessions）── 內嵌式結構**：
  ```json
  {
    "session_id": "sess_20260929_001",
    "user_id": "ricky",
    "turn_count": 2,
    "turns": [
      {
        "turn_id": 1,
        "query": "感測器如何安裝在樹幹上？",
        "response": "根據手冊說明，需使用隨附的金屬束帶固定於離地 1.5 公尺處...",
        "latency_ms": 312.4
      }
    ]
  }
  ```

* **稽核日誌（search_audit_logs）── 時序自動淘汰**：
  ```json
  {
    "timestamp": "2026-09-29T09:00:00Z",
    "query": "感測器如何安裝在樹幹上？",
    "vector_latency_ms": 45.1,
    "keyword_latency_ms": 12.3,
    "rerank_latency_ms": 120.5,
    "total_latency_ms": 177.9,
    "status": "success"
  }
  ```
  *(透過 `expireAfterSeconds: 2592000` 設定 30 天後自動刪除)*

### 5. 模組架構與檔案職責
* **[nosql/mongo_client.py](file:///Users/rickyho/Documents/github/scanPdf/nosql/mongo_client.py)**：
  * 單例模式管理連線池（`maxPoolSize=20`、`minPoolSize=2`），防範吃滿連線。
  * `init_indexes()` 自癒函式：自動建立會話複合索引（`user_id` + `updated_at`）與 30 天原生物理過期 TTL 索引（`idx_ttl_30d`）。
  * 支援斷線無感自動降級，連線異常時不影響主檢索服務。
* **[nosql/models.py](file:///Users/rickyho/Documents/github/scanPdf/nosql/models.py)**：
  * Pydantic 資料結構治理層（`ChatTurnModel`、`ChatSessionModel`、`SearchAuditLogModel`），統一 UTC 時間與數值邊界，杜絕動態綱要之髒資料。
* **[nosql/session_logger.py](file:///Users/rickyho/Documents/github/scanPdf/nosql/session_logger.py)**：
  * 採用雙背景執行緒池（`ThreadPoolExecutor`）進行無阻塞非同步寫入，Gradio 前端零延遲。
  * 單文件高內聚原子更新：`$push` 搭配 `$slice: -50`，嚴格限制每個會話最多內嵌 50 輪問答，杜絕 16 MB 上限溢出。
* **[nosql/analytics_pipeline.py](file:///Users/rickyho/Documents/github/scanPdf/nosql/analytics_pipeline.py)**：
  * 封裝四組企業級聚合管線：
    1. 檢索模式平均與最大耗時分組統計（`$group` + `$avg` + `$project` + `$round`）。
    2. 熱門搜尋問題詞彙頻次與延遲排序（`$group` + `$sort` + `$limit`）。
    3. 會話輪次結構重塑與摘要（`$project`）。
    4. 使用者滿意度星級分佈（`$unwind` + `$match`）。

### 6. MongoDB Atlas 雲端 M0 免費版限制與防禦矩陣
* **免安裝本機服務**：目前專案已直連 MongoDB Atlas 雲端叢集（`mongodb+srv://...`），無需在 macOS 下載或運行本機資料庫。
* **核心配額與防禦機制**：
  | 雲端配額項目 | M0 限制門檻 | 系統標準防禦機制 |
  | :--- | :--- | :--- |
  | **儲存容量** | 512 MB 上限 | 向量（Embedding）嚴格留存於 PostgreSQL；MongoDB 僅存輕量日誌，並以 30 天 TTL 索引自動非同步物理回收。 |
  | **併發連線** | 500 連線限制 | Python 驅動程式配置 `maxPoolSize=20` 保守連線池。 |
  | **網路頻寬** | 7 天 10 GB 限制 | 僅傳輸輕量 JSON，嚴禁全量拉取備份。 |
  | **單文件大小** | 16 MB 限制 | `$slice: -50` 視窗截斷，單一會話文件體積不超過 100 KB。 |

### 7. 測試與面試展示指令
* **驗證連線與索引自癒**：
  ```bash
  uv run python nosql/mongo_client.py
  ```
* **執行求職面試級指標聚合分析展示**：
  ```bash
  uv run python scripts/demo_mongodb_interview.py
  ```

---

## To do
1. Redis for memory
2. [x] MongoDB for chat sessions & search audit logging（已實作完成，整合多模型持久化與聚合管線）

## 部署到huggingface space 方法
- uv run python scripts/deploy_space.py


