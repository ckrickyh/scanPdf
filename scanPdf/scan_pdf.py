import os
import sys
from pathlib import Path
from paddleocr import PaddleOCRVL

def scan_pdf_with_ppocr_vl(pdf_path: str, output_dir: str = "output"):
    """
    使用 PaddleOCR-VL-1.6 視覺語言模型掃描並解析 PDF 文件。
    """
    target_file = Path(pdf_path)
    if not target_file.exists():
        print(f"錯誤：找不到目標檔案 {target_file.resolve()}", file=sys.stderr)
        sys.exit(1)

    # 確保輸出目錄存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"開始初始化 PaddleOCR-VL-1.6 文件解析管線...")
    print(f"==================================================")
    
    # 建立 PaddleOCR-VL-1.6 實例
    pipeline = PaddleOCRVL(pipeline_version="v1.6")

    print(f"\n正在讀取並掃描檔案：{target_file.name}")
    print(f"完整路徑：{target_file.resolve()}")
    
    # 執行模型推論解析
    output = pipeline.predict(str(target_file))

    print(f"\n==================================================")
    print(f"掃描完成，開始處理各頁面輸出結果...")
    print(f"==================================================")

    for index, res in enumerate(output):
        print(f"\n[頁面 {index + 1}] 解析預覽：")
        res.print()
        
        # 儲存結構化資料至輸出目錄
        res.save_to_json(save_path=str(output_path))
        res.save_to_markdown(save_path=str(output_path))

    print(f"\n全部處理完成！結構化 Markdown 與 JSON 已儲存至：{output_path.resolve()}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 預設指定原始手冊路徑與輸出目錄
    source_pdf = current_dir / "source" / "PiCUSQ72Manual.pdf"
    output_dir = project_root / "output"
    scan_pdf_with_ppocr_vl(pdf_path=str(source_pdf), output_dir=str(output_dir))
