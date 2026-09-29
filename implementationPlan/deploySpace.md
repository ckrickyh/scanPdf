# 技術手冊 RAG 線上展示（Hugging Face Spaces + Supabase）與一鍵同步部署實施規範

本文件定義將技術手冊 RAG 系統發布至 Hugging Face Spaces 雲端展示平台，並串接線上 Supabase pgvector 資料庫之架構設計、安全性隔離與一鍵自動化同步部署流程。

---

## 一、 設計目標與核心策略

1. **單一真實來源（Single Source of Truth）**：
   以專案根目錄之 `app.py` 作為唯一開發與展示核心，避免在本機展示與雲端部署間維護兩套分散程式碼。
2. **重型依賴徹底隔離**：
   本機具備 PaddleOCR 與 PaddlePaddle 等離線套件（總計超過 3 GB）；線上展示端僅保留檢索與 LLM 互動所需的極致輕量依賴（約 500 MB），大幅縮短雲端建置時間至 2 分鐘內。
3. **推論環境自適應降級**：
   * 本機模式：優先連線本機 Ollama 進行 GPU / Apple Silicon 向量推論。
   * 雲端模式：無 Ollama 守護程序時，自動降級調用 `sentence-transformers`（`BAAI/bge-m3`）於 CPU 進行單句問題編碼。
4. **連線池穩定性防護（Session Pooler 優先）**：
   採用 Supabase Session Pooler（Port 5432）維持連線生命週期，確保 `psycopg` 與 `pgvector` 自訂型態解碼器註冊不受 PgBouncer 交易模式干擾。
5. **一鍵鏡像同步工具**：
   建立自動化腳本，自動過濾離線管線檔案、生成 Spaces 專屬設定，並以安全方式同步至遠端。

---

## 二、 雙軌展示架構圖

```
【本機環境 (Local Environment)】                      【雲端展示環境 (Hugging Face Spaces)】
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│ 1. 離線切塊與入庫 Pipeline        │                 │ 1. 線上問答 UI (Gradio)         │
│    (pipeline_postgres.py)       │                 │    (deploy_space/app.py)        │
│    - PaddleOCR / Markdown       │                 │                                 │
│    - Ollama bge-m3 向量計算      │                 │ 2. 單句提問向量化                │
│                 │               │                 │    - sentence-transformers      │
│                 ▼               │                 │      (BAAI/bge-m3 / CPU 推論)   │
│ 2. 本機即時展示 (app.py)         │   一鍵同步工具   │                 │               │
│    - 支援 Ollama 高速推論       │ ──────────────> │                 ▼               │
│    - 支援 Gradio share 臨時外網  │ (deploy_space)  │ 3. Cross-Encoder 深度重排序     │
│                                 │                 │    (bge-reranker-v2-m3)         │
└────────────────┬────────────────┘                 └────────────────┬────────────────┘
                 │                                                   │
                 │ 寫入/讀取 (DB_TARGET=supabase)                      │ 唯讀檢索 (Port 5432)
                 ▼                                                   ▼
       ┌───────────────────────────────────────────────────────────────────────┐
       │                   線上 Supabase PostgreSQL (pgvector)                  │
       │                   - Table: manual_chunks                              │
       │                   - Index: HNSW (1024 維餘弦相似度) / GIN (BM25 全文)   │
       └───────────────────────────────────────────────────────────────────────┘
```

---

## 三、 資料庫連線與安全隔離規範

### 1. 連線協定與埠位決策
* **指定埠位：Session Pooler（Port 5432）**
* **網址格式**：
  ```
  postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require
  ```
* **技術決策依據**：
  Hugging Face Spaces 屬於常駐型容器，非 Serverless 短暫執行架構。`pgvector.psycopg.register_vector` 需要向資料庫註冊自訂二進位轉譯器，Transaction Pooler（Port 6543）在每筆交易結尾即清空 Session 狀態，極易引發預備陳述式衝突或型態轉換例外。

### 2. 金鑰安全防護（禁止提交 .env）
* 實體檔案 `.env` 必須自 Git 索引完全移除並加入 `.gitignore`。
* 本機讀取：依賴根目錄之 `.env`。
* 雲端讀取：由 Hugging Face Spaces 控制台之「Settings」->「Variables and secrets」以環境變數注入：
  * `DB_TARGET` = `supabase`
  * `SUPABASE_PG_CONN_STRING` = 完整連線字串
  * `GEMINI_API_KEY` = Google Gemini API 金鑰
  * `GEMINI_MODEL` = `gemini-2.5-flash`
  * `EMBEDDING_MODEL` = `bge-m3`

---

## 四、 輕量化部署目錄結構（.deploy_space/）

執行一鍵同步腳本時，系統將自動於本機建置暫存鏡像目錄 `.deploy_space/`：

```text
.deploy_space/
├── app.py                   # 根目錄 app.py 鏡像
├── chunkingRAG/             # 檢索必要模組（自動排除離線入庫程式）
│   ├── __init__.py
│   ├── db_config.py         # 資料庫連線配置
│   └── reranker.py          # Cross-Encoder 重排序模型服務
├── requirements.txt         # 線上專屬精簡依賴清單
└── README.md                # Hugging Face Spaces 專用中繼標頭檔
```

### 線上依賴清單定義（requirements.txt）
排除所有 PaddleOCR、LangChain 切分器等離線相依套件：
```text
gradio>=6.26.0
google-genai>=2.22.0
psycopg[binary]>=3.3.5
pgvector>=0.5.0
sentence-transformers>=6.0.1
torch
python-dotenv>=1.2.3
```

### Spaces 專用設定檔（README.md）
```markdown
---
title: Technical Manual RAG
emoji: 📑
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Technical Manual RAG System
技術手冊雙路混合檢索、Cross-Encoder 語意重排與 Google Gemini 綜合問答展示系統。
```

---

## 五、 一鍵同步部署腳本（scripts/deploy_space.py）規格

自動化腳本需具備以下作業階段：

1. **環境安全檢查（Pre-flight Validation）**：
   * 檢驗 `.env` 是否被排除於 Git 暫存區之外。
   * 檢驗 `app.py`、`chunkingRAG/db_config.py`、`chunkingRAG/reranker.py` 是否存在。
2. **目錄鏡像打包（Staging）**：
   * 清理並重建 `.deploy_space/` 目錄。
   * 複製 `app.py` 至 `.deploy_space/app.py`。
   * 建立 `.deploy_space/chunkingRAG/` 並僅複製執行期必要檔案（`__init__.py`、`db_config.py`、`reranker.py`）。
   * 寫入 Spaces 專用 `requirements.txt` 與 `README.md`。
3. **推送至 Hugging Face Spaces**：
   * 支援 `--dry-run` 參數供本機檢驗打包成果。
   * 支援透過 Git Subtree 自動推播：
     ```bash
     git subtree push --prefix .deploy_space [REMOTE_NAME] main
     ```

---

## 六、 本機雙軌展示與臨時外網分享功能

為兼顧本機展示需求，[app.py](file:///Users/rickyho/Documents/github/scanPdf/app.py) 增加動態環境變數判斷：

```python
# app.py 啟動配置
import os

share_enabled = os.getenv("GRADIO_SHARE", "false").strip().lower() == "true"

if __name__ == "__main__":
    import threading
    threading.Thread(target=RerankerService.get_model, daemon=True).start()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_enabled,
        css=custom_css
    )
```

* **本機純內網展示**：直接執行 `uv run python app.py`。
* **本機臨時外網展示**：執行 `GRADIO_SHARE=true uv run python app.py`，即刻取得 72 小時有效之公共連結。

---

## 七、 執行驗證清單（Checklist）

- [ ] 執行 `git rm -f --cached .env` 確保敏感金鑰移出 Git 暫存。
- [ ] 驗證 [.gitignore](file:///Users/rickyho/Documents/github/scanPdf/.gitignore) 已包含 `.env` 與 `.deploy_space/`。
- [ ] 於 Supabase SQL Editor 執行 [supabase_init.sql](file:///Users/rickyho/Documents/github/scanPdf/supabase_init.sql) 完成遠端資料庫結構就緒。
- [ ] 本機切換 `.env` 之 `DB_TARGET=supabase`，執行入庫管線：
  ```bash
  uv run python chunkingRAG/pipeline_postgres.py --input-dir output
  ```
- [ ] 於 Supabase 控制台確認 `manual_chunks` 筆數與 1024 維向量皆已完整入庫。
- [ ] 執行部署腳本產生 Staging 結構：
  ```bash
  uv run python scripts/deploy_space.py --dry-run
  ```
- [ ] 於 Hugging Face 建立 Space 並設定 Secrets。
- [ ] 執行一鍵推播，驗證雲端容器建置與檢索展示功能。
