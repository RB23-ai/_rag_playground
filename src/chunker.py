from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

class DocumentChunker:
    def __init__(self):
        # 1. Fixed-size: Simple and blunt
        self.fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        # 2. Recursive: Smartly splits by paragraph -> sentence -> word
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def chunk_fixed(self, text):
        return self.fixed_splitter.split_text(text)

    def chunk_recursive(self, text):
        return self.recursive_splitter.split_text(text)

    def chunk_semantic(self, text):
        # Simple heuristic: Split by double newlines as a 'semantic' stand-in 
        # or use a more advanced strategy. For this project, we'll use a 
        # Sentence Splitter which mimics semantic boundaries.
        sentences = text.split(". ")
        return [". ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]

    def run(self, docs, strategy="recursive"):
        chunks = []
        for doc in docs:
            content = doc["content"]
            if strategy == "fixed":
                parts = self.chunk_fixed(content)
            elif strategy == "semantic":
                parts = self.chunk_semantic(content)
            else:
                parts = self.chunk_recursive(content)
            
            for part in parts:
                chunks.append({"content": part, "metadata": doc["metadata"]})
        return chunks