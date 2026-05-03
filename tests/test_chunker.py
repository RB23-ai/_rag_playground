import pytest
from src.chunker import DocumentChunker

def test_recursive_chunking_logic():
    # Setup
    chunker = DocumentChunker()
    sample_text = "This is a sentence. This is another sentence. Here is a third."
    docs = [{"content": sample_text, "metadata": {"source": "test.pdf"}}]
    
    # Act: Use recursive strategy
    chunks = chunker.run(docs, strategy="recursive")
    
    # Assert
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "test.pdf"
    assert "content" in chunks[0]

def test_fixed_chunking():
    chunker = DocumentChunker()
    # Create a long string of 1000 'A's
    docs = [{"content": "A" * 1000, "metadata": {"source": "long.pdf"}}]
    
    chunks = chunker.run(docs, strategy="fixed")
    
    # Assert: Should have split into at least 2 chunks
    assert len(chunks) >= 2