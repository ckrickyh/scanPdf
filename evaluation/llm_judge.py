#!/usr/bin/env python3
"""技術手冊 RAG 問答品質評估模組（LLM-as-a-Judge）

本模組使用 Google Gemini 作為獨立評判者（Judge），針對檢索到的 Chunks 與生成答案進行多維度品質評估：
1. 脈絡相關性（Context Relevance）
2. 忠實度（Faithfulness）
3. 答案相關性（Answer Relevance）
4. 完整度（Completeness）

支援單筆評估與搭配 search_manual 進行端到端 RAG 評估。
執行方式：
    uv run evaluation/llm_judge.py
    uv run evaluation/llm_judge.py --query "感測器如何安裝在樹幹上？" --top-k 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 注入專案根目錄至 sys.path，確保由任何路徑執行皆可正常載入 chunkingRAG 核心模組
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from chunkingRAG.db_config import (
    get_gemini_api_key,
    get_gemini_model,
)
from chunkingRAG.search_manual import search_manual


class MetricJudgment(BaseModel):
    reasoning: str = Field(
        description="詳細評判推導過程。必須先列出具體文字依據與扣分原因，再給出分數。"
    )
    score: int = Field(
        ge=1,
        le=5,
        description="評分（整數 1 至 5 分，5 分為最優）",
    )


class JudgeReport(BaseModel):
    context_relevance: MetricJudgment = Field(
        description="檢索脈絡相關性：檢索到的切塊是否切中問題，有無無關雜訊"
    )
    faithfulness: MetricJudgment = Field(
        description="答案忠實度：生成答案是否嚴格基於檢索內容，有無未提及之事實或幻覺"
    )
    answer_relevance: MetricJudgment = Field(
        description="答案相關性：生成答案是否正面回應使用者問題的核心，有無答非所問"
    )
    completeness: MetricJudgment = Field(
        description="答案完整度：是否完整涵蓋檢索資料內可回答該問題的所有關鍵要素"
    )
    verdict: Literal["PASS", "FAIL"] = Field(
        description="整體判定（所有指標皆 >= 4 分時為 PASS，否則為 FAIL）"
    )
    summary_critique: str = Field(
        description="評審精煉總結與改善建議"
    )


class RAGJudge:
    def __init__(self, judge_model: str | None = None, max_retries: int = 4, base_delay: float = 2.0):
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("未在 .env 檢測到有效的 GEMINI_API_KEY 或 GOOGLE_API_KEY")

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.judge_model = judge_model or get_gemini_model()
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _call_gemini_with_retry(self, prompt: str) -> str:
        """具備抗網路重置、503 容量不足與模型自動降級之 API 呼叫防護"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeReport,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        candidate_models = [self.judge_model]
        for fallback in ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_err = None
        for current_model in candidate_models:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config,
                    )
                    if response.text:
                        if current_model != self.judge_model:
                            print(f"[提示] 主評判模型繁忙或受限，已自動切換至備援模型（{current_model}）完成評審。")
                        return response.text
                    raise RuntimeError("Gemini 回傳空白內容")
                except Exception as e:
                    last_err = e
                    err_msg = str(e).lower()
                    if any(kw in err_msg for kw in ["quota exceeded", "quota", "resource_exhausted", "limit:"]):
                        print(f"[提示] 模型 {current_model} 配額已飽和，立即切換至備援模型...")
                        break

                    is_transient = any(
                        kw in err_msg
                        for kw in [
                            "connection reset",
                            "broken pipe",
                            "remote disconnected",
                            "socket",
                            "timeout",
                            "deadline",
                            "503",
                            "unavailable",
                            "capacity",
                        ]
                    )
                    if not is_transient:
                        break
                    wait_time = self.base_delay * (2 ** (attempt - 1))
                    print(f"[警告] 模型 {current_model} 呼叫受阻（{e}），第 {attempt} 次重試，將於 {wait_time:.1f} 秒後重新發起...")
                    time.sleep(wait_time)

        raise last_err

    def evaluate(
        self,
        query: str,
        retrieved_contexts: list[str],
        generated_answer: str,
    ) -> JudgeReport:
        context_blocks = "\n\n---\n\n".join(
            [f"[Chunk {idx + 1}]\n{ctx}" for idx, ctx in enumerate(retrieved_contexts)]
        )

        prompt = f"""你是一位極度嚴謹的 RAG 系統品質裁判。請依據評分規約（Rubric）評判下列問答：

[使用者問題]
{query}

[檢索到的參考切塊 (Contexts)]
{context_blocks}

[受測系統生成答案 (Answer)]
{generated_answer}

[評分規約 (Rubric)]
1. 脈絡相關性 (Context Relevance):
   - 1 分: 檢索切塊完全與問題無關。
   - 3 分: 包含部分相關背景，但大量切塊為無關雜訊。
   - 5 分: 檢索切塊高精度命中問題所需資訊，無多餘無效資訊。

2. 忠實度 (Faithfulness / 幻覺檢測):
   - 1 分: 包含重大編造事實或與切塊內容直接矛盾。
   - 3 分: 核心論點符合，但夾帶未被切塊提及的外部假設或推測。
   - 5 分: 答案中的每一個事實與步驟，百分之百能從切塊精準推導。

3. 答案相關性 (Answer Relevance):
   - 1 分: 嚴重答非所問或離題。
   - 3 分: 僅回答次要問題，忽略核心提問。
   - 5 分: 清楚切中問題核心，邏輯緊密清晰。

4. 答案完整度 (Completeness):
   - 1 分: 遺漏切塊內提供的絕大多數關鍵步驟或安全限制。
   - 3 分: 回答了主要輪廓，但遺漏了重要的操作細節或數值規格。
   - 5 分: 切塊內所有可解答該問題的重要細節與操作限制皆完整包含。

[判定規則]
- reasoning 中必須先寫出具體扣分依據與文字對照，最後才給出 score。
- 只要任一指標 score < 4，verdict 必須為 FAIL；四個指標皆 >= 4 才可評為 PASS。
"""
        raw_json = self._call_gemini_with_retry(prompt)
        return JudgeReport.model_validate_json(raw_json)


def generate_candidate_answer(query: str, chunks: list[dict], model: str) -> str:
    """使用受測模型依據 Chunks 生成回答（含 503 自動降級與重試防護）"""
    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)
    ctx_text = "\n\n".join([f"【參考切塊 {i+1}】\n" + c["content"] for i, c in enumerate(chunks)])
    prompt = f"""請嚴格根據以下參考文件以繁體中文回答問題：

{ctx_text}

問題：{query}
回答："""

    candidate_models = [model]
    for fallback in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    for current_model in candidate_models:
        for attempt in range(1, 4):
            try:
                res = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                if res.text:
                    if current_model != model:
                        print(f"[提示] 受測生成模型繁忙，已降級使用備援模型（{current_model}）生成回答。")
                    return res.text.strip()
            except Exception as e:
                time.sleep(1.5 * attempt)

    return "無法生成回答（所有候選模型皆超載）。"


def main():
    parser = argparse.ArgumentParser(description="技術手冊 RAG LLM-as-a-Judge 評估工具")
    parser.add_argument("--query", "-q", default="感測器如何安裝在樹幹上？", help="測試問題")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="檢索切塊數量")
    parser.add_argument("--judge-model", default=None, help="評判模型（預設為環境變數設定之模型）")
    parser.add_argument("--candidate-model", default=None, help="受測生成模型（預設為環境變數設定之模型）")
    args = parser.parse_args()

    target_model = args.candidate_model or get_gemini_model()
    judge_model = args.judge_model or target_model

    print(f"\n{'='*65}")
    print(f"啟動 RAG LLM-as-a-Judge 自動化品質評審")
    print(f"測試問題：{args.query}")
    print(f"受測模型：{target_model} | 評判模型：{judge_model}")
    print(f"{'='*65}\n")

    # 1. 執行混合檢索
    print("階段 1：執行檢索與切塊召回...")
    results = search_manual(
        query=args.query,
        top_k=args.top_k,
        use_hybrid=True,
        use_reranker=True,
    )
    if not results:
        print("檢索未獲得任何結果，無法進行後續評估。")
        sys.exit(1)

    # 2. 生成受測回答
    print("\n階段 2：呼叫受測模型生成解答...")
    answer = generate_candidate_answer(args.query, results, target_model)
    print(f"生成回答內容：\n{answer}\n")

    # 3. 執行裁判評估
    print("階段 3：呼叫 LLM-as-a-Judge 進行客觀評審...")
    contexts = [r["content"] for r in results]
    judge = RAGJudge(judge_model=judge_model)
    report = judge.evaluate(query=args.query, retrieved_contexts=contexts, generated_answer=answer)

    # 4. 輸出評審報告
    print(f"\n{'='*65}")
    print(f"LLM-as-a-Judge 評審報告（總體判定：{report.verdict}）")
    print(f"{'='*65}")
    print(f"1. 脈絡相關性 (Context Relevance) : {report.context_relevance.score}/5")
    print(f"   理由: {report.context_relevance.reasoning}")
    print(f"2. 答案忠實度 (Faithfulness)        : {report.faithfulness.score}/5")
    print(f"   理由: {report.faithfulness.reasoning}")
    print(f"3. 答案相關性 (Answer Relevance)    : {report.answer_relevance.score}/5")
    print(f"   理由: {report.answer_relevance.reasoning}")
    print(f"4. 答案完整度 (Completeness)        : {report.completeness.score}/5")
    print(f"   理由: {report.completeness.reasoning}")
    print(f"\n[綜合點評與改善方向]")
    print(f"{report.summary_critique}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
