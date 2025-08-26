from langchain_tavily import TavilySearch
from app.config import vectorstore
import os
import numpy as np
from typing import List

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search = TavilySearch(k=8, tavily_api_key=TAVILY_API_KEY)

def normalize_scores(scores: List[float], method: str = "min_max") -> List[float]:
    """
    Normalize scores into range [0,1].
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ("min_max", "z_score", "sigmoid")
    """
    if not scores:
        return []
    
    scores = np.array(scores)
    
    if method == "min_max":
        # Min-Max normalization: (x - min) / (max - min)
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return [1.0] * len(scores)
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    elif method == "z_score":
        # Z-score normalization, then map into [0,1] with sigmoid
        mean_score, std_score = scores.mean(), scores.std()
        if std_score == 0:
            return [0.5] * len(scores)
        z_scores = (scores - mean_score) / std_score
        return (1 / (1 + np.exp(-z_scores))).tolist()
    
    elif method == "sigmoid":
        # Apply sigmoid directly
        return (1 / (1 + np.exp(-scores))).tolist()
    
    return scores.tolist()

def calculate_relevance_score(content: str, query: str) -> float:
    """
    Simple keyword-based relevance scoring.
    Can be replaced by embedding similarity or other advanced methods.
    """
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    if not query_words:
        return 0.0
    
    # Proportion of query words that appear in content
    matching_words = query_words.intersection(content_words)
    relevance = len(matching_words) / len(query_words)
    
    return relevance

def rag_context(query: str, top_k: int = 3, local_weight: float = 0.2, web_weight: float = 0.8):
    """
    Improved RAG pipeline with score normalization and unified ranking.
    
    Args:
        query: User query string
        top_k: Number of results to return (default: 3)
        local_weight: Weight for local (vectorstore) results
        web_weight: Weight for web search results
    """
    
    # 1. Local vectorstore search (small k because dataset is small)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    
    # Process local results - similarity_search may return distance, convert to similarity
    local_results = []
    local_scores = []
    
    for doc, score in docs_with_scores:
        # Convert distance to similarity: 
        # If score ≤ 1 assume cosine distance → similarity = 1 - distance
        # Else assume euclidean distance → similarity = 1 / (1 + distance)
        similarity_score = 1 - score if score <= 1 else 1 / (1 + score)
        
        local_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "raw_score": similarity_score,
            "source_type": "local"
        })
        local_scores.append(similarity_score)

    # 2. Web search
    try:
        web_results_raw = web_search.invoke(query)
        web_results = []
        web_scores = []
        
        for r in web_results_raw.get("results", []):
            # Compute a simple relevance score (keyword-based)
            relevance_score = calculate_relevance_score(r["content"], query)
            
            web_results.append({
                "content": r["content"],
                "metadata": {"source": r["url"], "title": r.get("title", "")},
                "raw_score": relevance_score,
                "source_type": "web"
            })
            web_scores.append(relevance_score)
            
    except Exception as e:
        print(f"Web search failed: {e}")
        web_results = []
        web_scores = []

    # 3. Normalize scores for each source
    if local_scores:
        normalized_local_scores = normalize_scores(local_scores, method="min_max")
        for i, result in enumerate(local_results):
            result["normalized_score"] = normalized_local_scores[i]
    
    if web_scores:
        normalized_web_scores = normalize_scores(web_scores, method="min_max")
        for i, result in enumerate(web_results):
            result["normalized_score"] = normalized_web_scores[i]

    # 4. Apply weights to compute final score
    all_results = []
    
    for result in local_results:
        if "normalized_score" in result:
            result["final_score"] = result["normalized_score"] * local_weight
            all_results.append(result)
    
    for result in web_results:
        if "normalized_score" in result:
            result["final_score"] = result["normalized_score"] * web_weight
            all_results.append(result)

    # 5. Sort and return top-k results
    all_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = all_results[:top_k]
    
    return top_results

def rag_context_advanced(query: str, top_k: int = 3, 
                        similarity_threshold: float = 0.3,
                        diversity_factor: float = 0.1):
    """
    Advanced version with threshold filtering and diversity.
    
    Args:
        query: User query string
        top_k: Number of results to return
        similarity_threshold: Minimum acceptable final_score
        diversity_factor: Factor to ensure content diversity (0=no filter, 1=very strict)
    """
    
    # Get more results first
    initial_results = rag_context(query, top_k=10)
    
    # 1. Filter by threshold
    filtered_results = [
        r for r in initial_results 
        if r["final_score"] >= similarity_threshold
    ]
    
    # 2. Ensure diversity (avoid very similar results)
    diverse_results = []
    for result in filtered_results:
        is_diverse = True
        for existing in diverse_results:
            # Naive content similarity: word overlap ratio
            content_similarity = len(set(result["content"].lower().split()) & 
                                   set(existing["content"].lower().split())) / \
                                max(len(result["content"].split()), 1)
            
            if content_similarity > (1 - diversity_factor):
                is_diverse = False
                break
        
        if is_diverse:
            diverse_results.append(result)
            
        if len(diverse_results) >= top_k:
            break
    
    return diverse_results
