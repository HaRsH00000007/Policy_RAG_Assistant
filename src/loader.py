import os
from pathlib import Path
from typing import List
import PyPDF2


def load_documents(directory: str = "data/policies") -> List[dict]:
    """
    Load all documents from the policies directory.
    Supports PDF, TXT, and MD files.
    
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    documents = []
    policy_dir = Path(directory)
    
    if not policy_dir.exists():
        print(f"Warning: {directory} does not exist")
        return documents
    
    for file_path in policy_dir.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = load_pdf(file_path)
                elif file_path.suffix.lower() in [".txt", ".md"]:
                    text = load_text(file_path)
                else:
                    continue
                
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": file_path.name,
                            "type": file_path.suffix[1:]
                        }
                    })
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    
    return documents


def load_pdf(file_path: Path) -> str:
    """Extract text from PDF file."""
    text = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)


def load_text(file_path: Path) -> str:
    """Load text from TXT or MD file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()