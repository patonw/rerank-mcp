import os
from enum import StrEnum, auto
from math import inf
from typing import Annotated

from fastmcp import FastMCP
from flashrank import Ranker, RerankRequest
from difflib import SequenceMatcher
import jellyfish

RERANK_MODEL = os.environ.get("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_CACHE = os.environ.get("RERANK_CACHE", "/tmp")

mcp = FastMCP("rerank-mcp")
ranker = Ranker(model_name=RERANK_MODEL, cache_dir=RERANK_CACHE)


class Similarity(StrEnum):
    SEMANTIC = auto()
    """Semantic similarity using a cross-encoder embedding model"""

    SEQUENTIAL = auto()
    """Python's built-in difflib SequenceMatcher"""

    JACCARD = auto()
    """Jaccard similarity on trigrams"""

    JARO = auto()
    """Jaro edit similarity"""

    JARO_WINKLER = auto()
    """Jaro-Winkler edit similarity"""


def eval_rankings(
    query: str,
    docs: list[dict],
    metric: Similarity,
) -> list[dict]:
    query = query.lower()
    match metric:
        case Similarity.SEMANTIC:
            rerankrequest = RerankRequest(query=query, passages=docs)
            results = ranker.rerank(rerankrequest)
            results = [{**it, "score": it["score"].item()} for it in results]
        case Similarity.SEQUENTIAL:
            results = [
                {
                    **it,
                    "score": SequenceMatcher(None, query, it["text"].lower()).ratio(),
                }
                for it in docs
            ]

            results = sorted(results, key=lambda it: it["score"], reverse=True)
        case Similarity.JACCARD:
            results = [
                {
                    **it,
                    "score": jellyfish.jaccard_similarity(
                        query, it["text"].lower(), ngram_size=3
                    ),
                }
                for it in docs
            ]

            results = sorted(results, key=lambda it: it["score"], reverse=True)
        case Similarity.JARO:
            results = [
                {**it, "score": jellyfish.jaro_similarity(query, it["text"].lower())}
                for it in docs
            ]

            results = sorted(results, key=lambda it: it["score"], reverse=True)
        case Similarity.JARO_WINKLER:
            results = [
                {**it, "score": jellyfish.jaro_similarity(query, it["text"].lower())}
                for it in docs
            ]

            results = sorted(results, key=lambda it: it["score"], reverse=True)
    return results


@mcp.tool
def rerank_docs(
    query: Annotated[str, "The text to rank documents against"],
    docs: Annotated[list[dict], "Documents to rank"],
    min_score: Annotated[float, "Minimum score to allow"] = -inf,
    max_score: Annotated[float, "Maximum score to allow"] = inf,
    limit: Annotated[int, "Maximum number of documents to return"] = None,
    metric: Annotated[
        Similarity, "Similarity metric used to evaluate score for each document"
    ] = Similarity.SEMANTIC,
) -> list[dict]:
    """Scores and reorders structured documents by semantic similarity to the query"""
    results = eval_rankings(query, docs, metric)

    results = [it for it in results if min_score <= it["score"] <= max_score]

    if limit:
        return results[:limit]
    else:
        return results


@mcp.tool
def rerank_texts(
    query: Annotated[str, "The text to rank documents against"],
    texts: Annotated[list[str], "Text passages to rank"],
    min_score: Annotated[float, "Minimum score to allow"] = -inf,
    max_score: Annotated[float, "Maximum score to allow"] = inf,
    limit: Annotated[int, "Maximum number of documents to return"] = None,
    metric: Annotated[
        Similarity, "Similarity metric used to evaluate score for each document"
    ] = Similarity.SEMANTIC,
    no_score: Annotated[bool, "Return results as plain text with no score"] = False,
) -> list[dict | str]:
    """Scores and reorders unstructured texts by semantic similarity to the query"""

    docs = [dict(text=it) for it in texts]
    results = eval_rankings(query, docs, metric)
    results = [it for it in results if min_score <= it["score"] <= max_score]

    if no_score:
        results = [it["text"] for it in results]

    if limit:
        return results[:limit]
    else:
        return results


def main():
    mcp.run()


if __name__ == "__main__":
    import pprint

    query = "How to speedup LLMs?"
    passages = [
        {
            "id": 1,
            "text": "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
            "meta": {"additional": "info1"},
        },
        {
            "id": 2,
            "text": "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
            "meta": {"additional": "info2"},
        },
        {
            "id": 3,
            "text": "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
            "meta": {"additional": "info3"},
        },
        {
            "id": 4,
            "text": "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
            "meta": {"additional": "info4"},
        },
        {
            "id": 5,
            "text": "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
            "meta": {"additional": "info5"},
        },
    ]

    rerankrequest = RerankRequest(query=query, passages=passages)
    results = eval_rankings(query, passages, Similarity.SEMANTIC)
    pprint.pp(results)
