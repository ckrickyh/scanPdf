-- ==============================================================================
-- Supabase 資料庫初始化腳本（支援 pgvector 與 HNSW 餘弦相似度索引）
-- 請在 Supabase 控制台的 SQL Editor 中直接貼上並執行
-- ==============================================================================

-- 1. 啟用向量擴充套件
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 建立文件切塊儲存表
CREATE TABLE IF NOT EXISTS manual_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id VARCHAR(120) UNIQUE NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    chunk_type VARCHAR(20) NOT NULL, -- 'text' 或 'table'
    chunk_index INT NOT NULL,
    h1 TEXT,
    h2 TEXT,
    h3 TEXT,
    h4 TEXT,
    h5 TEXT,
    content TEXT NOT NULL,
    embedding vector(1024) NOT NULL, -- 對應 BAAI/bge-m3 輸出之 1024 維度
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(h1, '') || ' ' || coalesce(h2, '') || ' ' || coalesce(h3, '') || ' ' || coalesce(h4, '') || ' ' || coalesce(h5, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. 建立 HNSW 向量索引（最佳化餘弦相似度檢索）
CREATE INDEX IF NOT EXISTS idx_manual_chunks_hnsw 
ON manual_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. 建立 GIN 全文檢索索引（最佳化關鍵字 BM25 檢索）
CREATE INDEX IF NOT EXISTS idx_manual_chunks_tsv ON manual_chunks USING gin(tsv);

-- 5. 建立常規欄位輔助索引
CREATE INDEX IF NOT EXISTS idx_manual_chunks_source ON manual_chunks (source_file);
CREATE INDEX IF NOT EXISTS idx_manual_chunks_type ON manual_chunks (chunk_type);
