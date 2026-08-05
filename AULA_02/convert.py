import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
from docling.document_converter import DocumentConverter

pdf_path = "escrita_academica_ia.pdf"

converter = DocumentConverter()
result = converter.convert(pdf_path)

markdown_output = result.document.export_to_markdown()

output_path = Path("escrita_academica_ia.md")
output_path.write_text(markdown_output, encoding="utf-8")

