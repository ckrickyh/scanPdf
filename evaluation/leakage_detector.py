#!/usr/bin/env python3
"""技術手冊 RAG 外部知識洩漏稽核器（External Knowledge Leakage Detector）

本模組專門用於驗證受測答案是否嚴格受限於檢索到的 PDF 參考切塊（Context Isolation），
防範大型語言模型偷偷使用預訓練權重中的先驗知識（Parametric Memory Leakage）。

核心機制：
1. 原子事實分解（Atomic Claim Decomposition）：將答案拆解為不可分割之獨立陳述句。
2. 上下文蘊含比對（Context Entailment Verification）：逐句檢驗在 PDF 切塊中是否有明確文字依據。
3. 外部洩漏率量化（Leakage Rate）：計算非手冊依據之外部陳述佔比。
4. 結構化判定（PASS_ISOLATED vs FAIL_LEAKAGE_DETECTED）。

執行方式：
    uv run evaluation/leakage_detector.py
    uv run evaluation/leakage_detector.py --query "感測器如何安裝在樹幹上？" --top-k 3
    uv run evaluation/leakage_detector.py --test-builtin
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 注入專案根目錄至 sys.path，確保正常載入 chunkingRAG 核心模組
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from chunkingRAG.db_config import (
    get_gemini_api_key,
    get_gemini_model,
)
from chunkingRAG.search_manual import search_manual
from evaluation.llm_judge import generate_candidate_answer


class AtomicFactCheck(BaseModel):
    statement: str = Field(
        description="從答案中拆解出的單一原子事實陳述句"
    )
    is_supported_by_context: bool = Field(
        description="該陳述是否能完全且直接在檢索切塊中找到明確依據（True: 來自切塊，False: 外部洩漏）"
    )
    citation_quote: str = Field(
        description="若有依據，摘錄參考切塊原文對應文字；若無，詳細說明此為切塊未載之外部推測或先驗事實"
    )


class LeakageAuditReport(BaseModel):
    atomic_checks: list[AtomicFactCheck] = Field(
        description="答案逐句原子事實檢核清單"
    )
    total_claims: int = Field(
        description="拆解出之原子事實陳述總數"
    )
    leaked_claims: int = Field(
        description="未載於參考切塊之外部先驗陳述數量"
    )
    leakage_rate: float = Field(
        description="外部知識洩漏率 (leaked_claims / total_claims，範圍 0.0 至 1.0)"
    )
    verdict: Literal["PASS_ISOLATED", "FAIL_LEAKAGE_DETECTED"] = Field(
        description="判定結果：洩漏率為 0.0 時為 PASS_ISOLATED，大於 0.0 則判定為 FAIL_LEAKAGE_DETECTED"
    )
    audit_summary: str = Field(
        description="稽核員綜合剖析與隔離度改善建議"
    )


class ExternalKnowledgeLeakageDetector:
    def __init__(
        self,
        audit_model: str | None = None,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ):
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("未在 .env 檢測到有效的 GEMINI_API_KEY 或 GOOGLE_API_KEY")

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.audit_model = audit_model or get_gemini_model()
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _call_gemini_with_retry(self, prompt: str) -> str:
        """具備抗網路連線重置、503 超載與模型自動降級之 API 呼叫防護"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LeakageAuditReport,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        candidate_models = [self.audit_model]
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
                        if current_model != self.audit_model:
                            print(f"[提示] 原模型（{self.audit_model}）受限，已自動切換至備援模型（{current_model}）完成檢驗。")
                        return response.text
                    raise RuntimeError("Gemini 回傳空白內容")
                except Exception as e:
                    last_err = e
                    err_msg = str(e).lower()
                    # 若為單一模型的配額用盡，直接跳至下一個備援模型
                    if any(kw in err_msg for kw in ["quota exceeded", "quota", "resource_exhausted", "limit:"]):
                        print(f"[提示] 模型 {current_model} 免費配額已飽和，立即切換至備援模型...")
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
                    print(f"[警告] 稽核模型 {current_model} 呼叫受阻（{e}），第 {attempt} 次重試...")
                    time.sleep(wait_time)

        raise last_err

    def audit(
        self,
        query: str,
        retrieved_contexts: list[str],
        generated_answer: str,
    ) -> LeakageAuditReport:
        context_blocks = "\n\n---\n\n".join(
            [f"[參考切塊 {idx + 1}]\n{ctx}" for idx, ctx in enumerate(retrieved_contexts)]
        )

        prompt = f"""你是一位極度嚴苛的文件隔離安全稽核員。你的唯一職責是：揪出受測答案中「所有未記載於參考切塊中的外部先驗知識與臆測」。

[使用者提問]
{query}

[PDF 參考切塊]
{context_blocks}

[受測模型生成答案]
{generated_answer}

【審查核心原則】：
1. 封閉世界假設（Closed-World Assumption）：除了上述【PDF 參考切塊】內的文字外，外部世界的一切事實皆不存在。
2. 即使答案內的某些事實（如螺絲型號、標準工具、常規物理常識）在真實世界中是絕對正確的常理，只要它「未明確寫在參考切塊中」，就屬於【外部先驗知識洩漏】。
3. 同義詞轉述（例如將『固定於樹幹』寫為『安裝在樹身上』）屬於允許範疇，但若引入了新的數值、新零件、新規格或新條件，必須判定為洩漏。

【執行步驟】：
1. 將【受測模型生成答案】拆解為最小不可分割之獨立原子陳述句（statement）。
2. 對每一句進行嚴格檢查：
   - 若能完全在參考切塊找到文字依據，is_supported_by_context = true，citation_quote 填入原文短句。
   - 若無法在切塊找到直接依據，is_supported_by_context = false，citation_quote 填寫具體洩漏原因（例如：手冊未提及此工具規格）。
3. 計算 total_claims 與 leaked_claims，計算 leakage_rate = leaked_claims / total_claims。
4. 若 leaked_claims > 0，verdict 必須為 FAIL_LEAKAGE_DETECTED；僅當 leaked_claims == 0 時，verdict 始為 PASS_ISOLATED。
"""

        raw_json = self._call_gemini_with_retry(prompt)
        return LeakageAuditReport.model_validate_json(raw_json)


def run_builtin_tests():
    """執行內建對照測試（基準純淨樣本 vs 人為下毒洩漏樣本）"""
    print(f"\n{'='*65}")
    print("執行外部知識洩漏稽核器內建基準測試（Built-in Benchmark Test）")
    print(f"{'='*65}\n")

    detector = ExternalKnowledgeLeakageDetector()

    sample_query = "感測器安裝在樹幹上的具體步驟？"
    sample_context = [
        "安裝說明：感測器必須以專用金屬釘固定於樹幹表面，釘入深度約為 2 公分，確保探針緊貼樹皮木質部。"
    ]

    # 案例 A：完全純淨的答案
    pure_answer = "安裝感測器時，需使用專用金屬釘釘入樹幹約 2 公分，並確認探針緊貼樹皮木質部。"
    print("[測試案例 A：純淨無洩漏樣本]")
    report_a = detector.audit(sample_query, sample_context, pure_answer)
    print(f"判定結果：{report_a.verdict}（洩漏率：{report_a.leakage_rate * 100:.1f}%）")
    for idx, c in enumerate(report_a.atomic_checks, 1):
        tag = "合格" if c.is_supported_by_context else "洩漏"
        print(f"  [{idx}] [{tag}] {c.statement} -> {c.citation_quote}")

    # 案例 B：夾帶外部常識腦補的答案（夾帶手冊沒寫的通用螺栓規格與防鏽提示）
    leaked_answer = (
        "安裝感測器時，需使用專用金屬釘釘入樹幹約 2 公分使探針緊貼木質部。"
        "另外建議在釘孔周圍塗抹矽膠密封膠以防水，並使用 M6 不鏽鋼螺栓加固防鏽。"
    )
    print(f"\n[測試案例 B：外部先驗知識洩漏樣本（人為注入矽膠與 M6 螺栓）]")
    report_b = detector.audit(sample_query, sample_context, leaked_answer)
    print(f"判定結果：{report_b.verdict}（洩漏率：{report_b.leakage_rate * 100:.1f}%）")
    for idx, c in enumerate(report_b.atomic_checks, 1):
        tag = "合格" if c.is_supported_by_context else "違規洩漏"
        print(f"  [{idx}] [{tag}] {c.statement} -> {c.citation_quote}")
    print(f"\n稽核總評：{report_b.audit_summary}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="技術手冊 RAG 外部知識洩漏稽核器")
    parser.add_argument("--query", "-q", default="感測器如何安裝在樹幹上？", help="測試問題")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="檢索切塊數量")
    parser.add_argument("--model", "-m", default=None, help="稽核評審模型（預設為環境變數設定）")
    parser.add_argument("--test-builtin", action="store_true", help="執行內建對照測試（純淨樣本 vs 洩漏樣本）")
    args = parser.parse_args()

    if args.test_builtin:
        run_builtin_tests()
        return

    print(f"\n{'='*65}")
    print(f"啟動 RAG 外部知識洩漏稽核（External Knowledge Leakage Audit）")
    print(f"測試問題：{args.query}")
    print(f"{'='*65}\n")

    # 1. 檢索參考切塊
    print("階段 1：檢索技術手冊切塊...")
    results = search_manual(query=args.query, top_k=args.top_k, use_hybrid=True, use_reranker=True)
    if not results:
        print("檢索未獲得任何結果，無法進行後續稽核。")
        sys.exit(1)

    # 2. 生成受測答案
    target_model = get_gemini_model()
    print(f"\n階段 2：呼叫受測模型（{target_model}）生成回答...")
    answer = generate_candidate_answer(args.query, results, target_model)
    print(f"生成回答：\n{answer}\n")

    # 3. 執行知識洩漏稽核
    print("階段 3：執行原子事實拆解與外部知識洩漏深度稽核...")
    contexts = [r["content"] for r in results]
    detector = ExternalKnowledgeLeakageDetector(audit_model=args.model)
    report = detector.audit(query=args.query, retrieved_contexts=contexts, generated_answer=answer)

    # 4. 輸出稽核成果
    print(f"\n{'='*65}")
    print(f"外部知識洩漏稽核報告（最終判定：{report.verdict}）")
    print(f"{'='*65}")
    print(f"原子事實總數：{report.total_claims} | 外部洩漏陳述：{report.leaked_claims}")
    print(f"外部知識洩漏率：{report.leakage_rate * 100:.1f}%")
    print(f"\n[逐句原子陳述稽核清單]")
    for idx, c in enumerate(report.atomic_checks, 1):
        status_label = "[合格：文本依據]" if c.is_supported_by_context else "[違規：外部洩漏]"
        print(f"  {idx}. {status_label} {c.statement}")
        print(f"     依據/說明：{c.citation_quote}")

    print(f"\n[稽核員總評與診斷]")
    print(f"{report.audit_summary}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
