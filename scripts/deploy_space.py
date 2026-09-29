"""
Hugging Face Spaces 一鍵鏡像打包與自動化同步部署腳本
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT_DIR / ".deploy_space"

REQUIRED_SOURCE_FILES = [
    ROOT_DIR / "app.py",
    ROOT_DIR / "chunkingRAG" / "__init__.py",
    ROOT_DIR / "chunkingRAG" / "db_config.py",
    ROOT_DIR / "chunkingRAG" / "reranker.py",
]

SPACES_REQUIREMENTS = """gradio>=6.26.0
google-genai>=2.22.0
psycopg[binary]>=3.3.5
pgvector>=0.5.0
sentence-transformers>=6.0.1
torch
python-dotenv>=1.2.3
spaces
"""

SPACES_README = """---
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
"""


def preflight_check() -> None:
    """檢查必要檔案與 Git 安全狀態。"""
    print("==> [1/3] 執行環境安全與依賴完整性檢查 (Pre-flight Validation)...")
    
    # 檢查必要檔案
    for file_path in REQUIRED_SOURCE_FILES:
        if not file_path.exists():
            print(f"[Status: ERROR] 缺少必要來源檔案：{file_path.relative_to(ROOT_DIR)}")
            sys.exit(1)

    # 檢查 .env 是否被意外加入 Git 暫存
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", ".env"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if git_status and not git_status.startswith("??"):
            print("[Status: ERROR] 偵測到 .env 處於 Git 追蹤或暫存狀態，請先執行 `git rm --cached .env`！")
            sys.exit(1)
    except Exception:
        pass

    print("    依賴檔案與資安檢查通過。")


def stage_deploy_files() -> None:
    """建置輕量化部署目錄結構。"""
    print(f"==> [2/3] 建置輕量化鏡像目錄 ({DEPLOY_DIR.name}/)...")

    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 複製 app.py
    shutil.copy2(ROOT_DIR / "app.py", DEPLOY_DIR / "app.py")

    # 2. 複製 chunkingRAG 必要模組
    target_rag_dir = DEPLOY_DIR / "chunkingRAG"
    target_rag_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_DIR / "chunkingRAG" / "__init__.py", target_rag_dir / "__init__.py")
    shutil.copy2(ROOT_DIR / "chunkingRAG" / "db_config.py", target_rag_dir / "db_config.py")
    shutil.copy2(ROOT_DIR / "chunkingRAG" / "reranker.py", target_rag_dir / "reranker.py")

    # 3. 寫入 requirements.txt
    (DEPLOY_DIR / "requirements.txt").write_text(SPACES_REQUIREMENTS, encoding="utf-8")

    # 4. 寫入 README.md
    (DEPLOY_DIR / "README.md").write_text(SPACES_README, encoding="utf-8")

    print(f"    已生成目錄：{DEPLOY_DIR}")
    print("    包含檔案：")
    for p in DEPLOY_DIR.rglob("*"):
        if p.is_file():
            print(f"      - {p.relative_to(DEPLOY_DIR)}")


def push_to_remote(remote_name: str, branch: str = "main") -> None:
    """推送至遠端 Hugging Face Space（採用獨立 Staging Git 推送，避免依賴母專案 Git 樹狀追蹤）。"""
    print(f"==> [3/3] 推送 Staging 內容至遠端 {remote_name}/{branch}...")
    
    # 取得遠端 URL
    if remote_name.startswith("http://") or remote_name.startswith("https://") or remote_name.startswith("git@"):
        remote_url = remote_name
    else:
        try:
            remote_url = subprocess.check_output(
                ["git", "remote", "get-url", remote_name],
                cwd=ROOT_DIR,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            print(f"[Status: ERROR] 找不到 Git 遠端名稱 '{remote_name}'。")
            print(f"請先於母專案註冊 Hugging Face 遠端：")
            print(f"  git remote add {remote_name} https://huggingface.co/spaces/<使用者名稱>/<Space名稱>")
            sys.exit(1)

    print(f"    目標遠端 URL: {remote_url}")

    # 於 DEPLOY_DIR 內部獨立初始化 Git 並強制推送，確保 .deploy_space/ 保持於 .gitignore
    subprocess.run(["git", "init"], cwd=DEPLOY_DIR, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", "-B", branch], cwd=DEPLOY_DIR, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "add", "."], cwd=DEPLOY_DIR, check=True)
    subprocess.run(["git", "commit", "-m", "Deploy to Hugging Face Spaces"], cwd=DEPLOY_DIR, check=True, stdout=subprocess.DEVNULL)

    print(f"    執行指令: git push -f {remote_name} {branch}")
    result = subprocess.run(["git", "push", "-f", remote_url, branch], cwd=DEPLOY_DIR)
    if result.returncode != 0:
        print("[Status: ERROR] 推送至 Hugging Face Space 失敗，請確認網路與存取權限。")
        sys.exit(result.returncode)
    print("    部署同步成功！")


def main() -> None:
    parser = argparse.ArgumentParser(description="技術手冊 RAG - Hugging Face Spaces 一鍵打包與同步工具")
    parser.add_argument("--dry-run", action="store_true", help="僅生成 .deploy_space/ 目錄供檢驗，不執行遠端推送")
    parser.add_argument("--remote", type=str, default="space", help="Hugging Face Space 的 Git Remote 名稱（預設: space）")
    parser.add_argument("--branch", type=str, default="main", help="目標分支名稱（預設: main）")

    args = parser.parse_args()

    preflight_check()
    stage_deploy_files()

    if args.dry_run:
        print("\n[完成] --dry-run 模式已結束。已成功產生 .deploy_space/，未執行遠端推送。")
    else:
        push_to_remote(remote_name=args.remote, branch=args.branch)


if __name__ == "__main__":
    main()
