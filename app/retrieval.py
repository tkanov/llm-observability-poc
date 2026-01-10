import glob
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

logger = logging.getLogger(__name__)

# Global variables to store the index (cached for efficiency, built once and reused)
_vectorizer = None
_snippets = None
_source_ids = None
_tfidf_matrix = None  # Store pre-computed TF-IDF matrix for efficiency

# Get project root (parent of app directory)
_PROJECT_ROOT = Path(__file__).parent.parent


def _load_knowledge_base(kb_dir: str = "data/kb") -> List[tuple]:
    """
    Load all markdown files from the knowledge base directory.
    
    Args:
        kb_dir: Path to knowledge base directory (relative to project root)
    
    Returns:
        List of tuples: (source_id, full_text)
    """
    # Resolve path relative to project root
    kb_path = _PROJECT_ROOT / kb_dir
    if not kb_path.exists():
        logger.warning(f"Knowledge base directory {kb_path} does not exist")
        return []
    
    documents = []
    md_files = glob.glob(str(kb_path / "*.md"))
    
    for md_file in md_files:
        source_id = Path(md_file).stem  # Get filename without extension
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append((source_id, content))
                    logger.info(f"Loaded {source_id} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Error loading {md_file}: {e}")
    
    return documents


def _chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split a document into chunks for better retrieval.
    
    Args:
        text: Document text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary near the end to avoid cutting mid-sentence
        if end < len(text):
            # Look backwards from target end for sentence endings (., !, ?) or newlines
            # Range goes from end backwards, but don't go further than halfway through chunk
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?\n':
                    end = i + 1  # Include the punctuation in the chunk
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position back by overlap amount to create overlapping chunks
        # This ensures context is preserved across chunk boundaries
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def _build_index(kb_dir: str = "data/kb"):
    """
    Build TF-IDF index from knowledge base files.
    
    Args:
        kb_dir: Path to knowledge base directory
    """
    global _vectorizer, _snippets, _source_ids, _tfidf_matrix
    
    # Load documents
    documents = _load_knowledge_base(kb_dir)
    
    # Chunk documents into snippets
    snippets = []
    source_ids = []
    
    for source_id, text in documents:
        chunks = _chunk_document(text)
        for chunk in chunks:
            snippets.append(chunk)
            source_ids.append(source_id)
    
    # Initialize empty index if no snippets found
    if not snippets:
        logger.warning("No snippets generated from knowledge base")
        _vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        _snippets = []
        _source_ids = []
        _tfidf_matrix = None
        return
    
    # Build TF-IDF vectorizer and fit on snippets
    # ngram_range=(1, 2) creates features for both single words (unigrams) and word pairs (bigrams)
    _vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    _tfidf_matrix = _vectorizer.fit_transform(snippets)
    
    _snippets = snippets
    _source_ids = source_ids
    
    logger.info(f"Built index with {len(snippets)} snippets from {len(documents)} documents")


def _ensure_index():
    """Ensure the index is built before retrieval."""
    global _vectorizer, _tfidf_matrix
    
    if _vectorizer is None or _tfidf_matrix is None:
        logger.info("Building TF-IDF index...")
        _build_index()
    else:
        logger.debug("Reusing existing TF-IDF index")


def retrieve(customer_message: str) -> List[Dict[str, str]]:
    """
    Retrieve relevant snippets from knowledge base based on customer message.
    
    Args:
        customer_message: Customer message string
    
    Returns:
        List of snippet dicts with source_id and excerpt (top 3 by similarity)
    """
    start_time = time.time()
    logger.info(f"Starting retrieval for query: {customer_message[:100]}...")
    
    _ensure_index()
    
    if not _snippets or not _vectorizer or _tfidf_matrix is None:
        logger.warning("Index not available, returning empty results")
        return []
    
    logger.debug(f"Index contains {len(_snippets)} snippets to search")
    
    try:
        # Vectorize the query using the same vectorizer as the index
        query_vector = _vectorizer.transform([customer_message])
        logger.debug("Query vectorized successfully")
        
        # Calculate cosine similarity using pre-computed TF-IDF matrix
        # flatten() converts 2D array (1 query × N snippets) to 1D array of N similarities
        similarities = cosine_similarity(query_vector, _tfidf_matrix).flatten()
        
        # Get top 3 indices: argsort() gives indices sorted by similarity (ascending),
        # [-3:] takes last 3 (highest), [::-1] reverses to get descending order
        top_indices = similarities.argsort()[-3:][::-1]
        
        # Build results
        results = []
        filtered_count = 0
        for idx in top_indices:
            similarity_score = similarities[idx]
            # Only include snippets with non-zero similarity
            if similarity_score > 0:
                results.append({
                    "source_id": _source_ids[idx],
                    "excerpt": _snippets[idx]
                })
                logger.debug(f"Added snippet from {_source_ids[idx]} with similarity: {similarity_score:.4f}")
            else:
                filtered_count += 1
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        if results:
            logger.info(
                f"Retrieved {len(results)} snippets (filtered {filtered_count} zero-similarity) "
                f"in {elapsed_ms}ms for query: {customer_message[:50]}..."
            )
        else:
            logger.warning(
                f"No relevant snippets found (all {len(top_indices)} candidates filtered) "
                f"in {elapsed_ms}ms for query: {customer_message[:50]}..."
            )
        
        return results
        
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error during retrieval after {elapsed_ms}ms: {e}", exc_info=True)
        return []
