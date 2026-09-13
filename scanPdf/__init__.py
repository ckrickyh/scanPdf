"""
scanPdf 套件
提供基於 PaddleOCR-VL 之 PDF 文件結構化掃描與解析功能。

- 無 __init__.py, from scanPdf.scan_pdf import 
- 有 __init__.py, from scanPdf import scan_pdf_with_ppocr_vl
"""

from .scan_pdf import scan_pdf_with_ppocr_vl

__all__ = ["scan_pdf_with_ppocr_vl"]