"""
Document Text Extractor
Extracts text from uploaded PDF and text files.
"""

from pathlib import Path


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        from pdfminer.high_level import extract_text
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text = extract_text(tmp_path)
        os.unlink(tmp_path)
        return text.strip()
    except ImportError:
        return "[ERROR] pdfminer not installed. Run: pip install pdfminer.six"
    except Exception as e:
        return f"[ERROR] PDF extraction failed: {e}"


def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from uploaded text file."""
    try:
        raw = uploaded_file.read()
        raw = raw.replace(b"\x00", b"")
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"[ERROR] Text extraction failed: {e}"


def extract_text_from_pdf_path(pdf_path: str) -> str:
    """Extract text from a PDF file path on local disk."""
    try:
        from pdfminer.high_level import extract_text

        path = Path(pdf_path)
        if not path.exists():
            return f"[ERROR] PDF not found: {pdf_path}"
        text = extract_text(str(path))
        return text.strip()
    except ImportError:
        return "[ERROR] pdfminer not installed. Run: pip install pdfminer.six"
    except Exception as e:
        return f"[ERROR] PDF extraction failed: {e}"
