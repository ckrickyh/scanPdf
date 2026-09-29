from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field


class FeedbackModel(BaseModel):
    """使用者問答回饋模型"""
    rating: int = Field(ge=1, le=5, description="評分 1 至 5 分")
    comment: Optional[str] = Field(default=None, max_length=500, description="回饋備註")


class ChatTurnModel(BaseModel):
    """單一問答輪次模型（內嵌於會話中）"""
    turn_id: int = Field(ge=1, description="該會話內的輪次編號")
    query: str = Field(min_length=1, description="使用者提問內容")
    response: str = Field(description="模型回覆內容")
    retrieved_chunk_ids: List[str] = Field(default_factory=list, description="檢索召回之 Chunk ID 清單")
    latency_ms: float = Field(ge=0.0, description="端到端總耗時（毫秒）")
    feedback: Optional[FeedbackModel] = Field(default=None, description="使用者評分回饋")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="該輪對話發生時間（UTC）")


class ChatSessionModel(BaseModel):
    """使用者會話聚合模型（展示內嵌模型設計）"""
    session_id: str = Field(min_length=1, description="會話唯一識別碼")
    user_id: str = Field(default="anonymous", description="使用者識別碼")
    turn_count: int = Field(default=0, ge=0, description="累積對話輪次")
    turns: List[ChatTurnModel] = Field(default_factory=list, description="內嵌之多輪問答清單（上限 50 輪）")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="會話建立時間")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="最後更新時間")


class SearchAuditLogModel(BaseModel):
    """檢索稽核日誌模型（展示時序日誌與 TTL 自動清理）"""
    session_id: Optional[str] = Field(default=None, description="關聯之會話 ID")
    query: str = Field(min_length=1, description="檢索關鍵字句")
    query_type: str = Field(default="hybrid", description="檢索模式：hybrid, vector, keyword")
    chunk_type_filter: Optional[str] = Field(default=None, description="區塊篩選條件：text, table")
    vector_latency_ms: float = Field(default=0.0, ge=0.0, description="向量搜尋耗時（毫秒）")
    keyword_latency_ms: float = Field(default=0.0, ge=0.0, description="全文關鍵字搜尋耗時（毫秒）")
    rerank_latency_ms: float = Field(default=0.0, ge=0.0, description="Cross-Encoder 重排序耗時（毫秒）")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="檢索流程總耗時（毫秒）")
    result_count: int = Field(default=0, ge=0, description="召回結果筆數")
    top_scores: List[float] = Field(default_factory=list, description="前幾筆結果之相似度或 RRF 分數")
    status: str = Field(default="success", description="狀態：success, error")
    error_message: Optional[str] = Field(default=None, description="異常訊息")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="紀錄時間戳記（UTC）")
