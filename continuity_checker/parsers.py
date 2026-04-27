from pathlib import Path

import pdfplumber
from docx import Document


def _parse_pdf(file_path: str) -> str:
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())
    return "\n".join(pages_text)


def _parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)


def _parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext == ".doc":
        raise ValueError("不支持 .doc 格式，请将文件另存为 .docx 格式后重试。")
    elif ext == ".txt":
        return _parse_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .pdf、.docx、.txt 格式。")
