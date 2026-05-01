import os 
from pathlib import Path 
from pypdf import PdfReader

class PDFIngestor:
    def __init__(self, data_dir: str = "data"):
        self.data_path = Path(data_dir)
        
    def load_pdfs(self):
        documents = []
        pdf_files = list(self.data_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f" No PDFs found in {self.data_path}. Drop some files there!")
            return []

        for pdf_file in pdf_files:
            try:
                reader = PdfReader(pdf_file)
                text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                documents.append({"content": text, "metadata": {"source": pdf_file.name}})
                print(f" Loaded: {pdf_file.name}")
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")
        return documents

if __name__ == "__main__":
    docs = PDFIngestor().load_pdfs()
    print(f"Done! Ingested {len(docs)} documents.")