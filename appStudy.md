# Logic

## scanPdf
- Vision Language Model (VLM) 掃描 PDF，抽取文本與 HTML 表格資訊
- 自動修復 OCR 辨識瑕疵與文字對齊

## Embedding（資料入庫階段）
- 結構預處理：
  - Markdown 標題切分：依據 H1、H2、H3、H4、H5 階層保留章節元數據
  - 原生 HTML <table> 獨立抽取為完整表格區塊
- 語意切分（Semantic Chunking）：將長文本切分成具有完整語意的「文本區塊 / 段落（Chunks）」
  - 計算相鄰句子語意距離，以動態百分位數（Percentile=85）進行斷句
- 上下文重疊補償：
  - 以正規表達式（re）向前回溯前一個 Chunk 結尾之完整句子（目標約 250 字元 / 100-120 Tokens）銜接至次區塊，確保語意連貫
  - 若該句超過 200 Tokens，則向前保留約 200 Tokens 作為重疊前綴（使重疊內容佔比小於 15%，防止主句語意被稀釋，同時提升檢索連貫性）
- 向量計算（Embedding）：
  - 將已完成上下文拼接的「完整文字區塊」送入 Attention 網路（bge-m3 模型），輸出 1024 維稠密向量（Dense Vector）
- 資料庫存入（DB Ingestion）：
  - 將文本內容與向量寫入 PostgreSQL 的 manual_chunks 資料表
  - 建立 HNSW 向量索引：以餘弦距離（Cosine Distance）為度量尺規，將語意相近的句子節點互相連結，建立多層導航圖形網絡
- 兩階段檢索架構（Two-Stage Retrieval with Reranker）：
  - 第一階段粗篩（High Recall）：由向量檢索自資料庫撈出 15 至 20 筆候選區塊（Candidate Pool）。
  - 第二階段精篩（High Precision）：採用交叉編碼器（Cross-Encoder，`BAAI/bge-reranker-v2-m3`）進行 Query-Doc 一對一全注意力打分，重排並截取最高分的 Top-3 至 Top-5。

## LLM & Retrieval（檢索與問答階段）
- 問題向量化：提問問題經 bge-m3 模型轉換為 1024 維查詢向量
- 語意檢索（Semantic Search）：透過 HNSW 索引計算 Cosine Similarity，高效撈出最相關的前 K 筆候選區塊（Top-K）
- 答案生成（Deterministic Generation）：LLM（Google Gemini 2.5 Flash）綜合精篩後的 Top-K 參考文件，配置 `temperature=0.2` 與 `top_p=0.85`，壓制幻覺機率，產出高事實性繁體中文解答。


## Evaluate RAG（檢索指標評估）
- Recall@K : TP / (TP + FN)（衡量真實目標答案被找回的比例）
- Precision@K : TP / (TP + FP)（衡量回傳結果中有效資訊的純度）
- F1-score : 2 * (Precision * Recall) / (Precision + Recall)（精確率與召回率之調和平均數 Harmonic Mean）
- 指標策略：RAG 檢索核心任務為「寧可多抓、不可漏抓」，通常以 Recall@K 與 Hit Rate@K 為主要驗收指標；固定 K 值會人為壓低 Precision 與 F1-score。

## audit 後問 
- 使用 uvx 調用安全審計工具，掃描專案環境所有套件是否有已知後門與 CVE 漏洞
- uvx pip-audit

## 相似度過濾門檻
- 加入 min_similarity >= 0.5 條件，驗證無關提問（如義大利披薩）精確回傳 0 筆，阻斷雜訊干擾。

## reranker
- 套件就位：透過 uv add sentence-transformers 完成 PyTorch 2.14.0 與 Hugging Face 函式庫安裝。
重排核心實作：建立 
- reranker.py
 封裝 BAAI/bge-reranker-v2-m3，啟用 Apple Silicon MPS 晶片加速，實作 Sigmoid 正規化。
- 檢索流程升級：資料庫粗篩 15 筆候選池（High Recall）-> Cross-Encoder 逐對計算交叉注意力並重排精選 Top-3（High Precision）。

## 進化 - vector 再結合 tsvector 

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

## Top-K vs Top-P（賽跑頒獎 vs 錄取及格）
- Top-K（看名次）：比賽「永遠只頒獎給前 3 名（K=3）」。不管第四名只差 0.01 秒，還是第一名贏第二名十條街，永遠只抓剛好 3 個人。
- Top-P（看實力佔比）：面試決定「錄取累積實力達到全體前 85%（P=0.85）的人」。若榜首一個人實力就獨佔 90%，這梯次就只錄取他 1 個人；若大家實力平分秋色，就可能一口氣錄取 8 個人。

## Test
uv run python chunkingRAG/search_manual.py --query "How to mount the sensor on tree?" --rerank
