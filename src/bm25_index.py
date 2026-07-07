import math
import re
from collections import Counter
from typing import Any, Callable, Optional

from hybrid_retriever import validate_document


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
    ) -> None:
        self.documents: list[dict[str, Any]] = []
        self._tokenized_docs: list[list[str]] = []
        self._doc_len: list[int] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def add_document(self, document: dict[str, Any]) -> None:
        validate_document(document)

        doc_tokens = self._tokenizer(document["content"])
        self._tokenized_docs.append(doc_tokens)

        self.documents.append(document)
        self._update_doc_stats(doc_tokens)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return

        for i, document in enumerate(documents):
            validate_document(document=document, index=i)

        for document in documents:
            doc_tokens = self._tokenizer(document["content"])
            self._tokenized_docs.append(doc_tokens)
            self.documents.append(document)
            self._update_doc_stats(doc_tokens)

    def clear(self) -> None:
        self.documents = []
        self._tokenized_docs = []
        self._doc_len = []
        self._doc_freqs = {}
        self._avg_doc_len = 0.0
        self._idf = {}
        self._index_built = False

    def _update_doc_stats(self, doc_tokens: list[str]) -> None:
        # Record this document's token count
        self._doc_len.append(len(doc_tokens))

        # Count how many documents contain each unique token
        for token in set(doc_tokens):
            self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Mark index as stale so IDF is recomputed before next search
        self._index_built = False
        ...

    def _calculate_idf(self) -> None:
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self) -> None:
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return
        # Compute average document length across the corpus
        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        # Compute IDF score for each term from _doc_freqs
        self._calculate_idf()
        # Mark index as up to date
        self._index_built = True

    def _compute_bm25_score(self, query_tokens: list[str], doc_index: int) -> float:
        score = 0.0

        # Count how many times each token appears in this document
        doc_term_counts = Counter(self._tokenized_docs[doc_index])
        # Get this document's length
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            # Skip if the token has no IDF (not in any document)
            if token not in self._idf:
                continue

            # Look up its IDF and term frequency in this document
            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            # BM25 scoring: combines rarity (IDF) with term frequency, capped by k1 and adjusted for document length
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            # Add to total score
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self, query: str, k: int = 1, score_normalization_factor: float = 0.1
    ) -> list[tuple[dict[str, Any], float]]:
        if not self.documents:
            return []

        if not isinstance(query, str):
            raise TypeError("Query text must be a string.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Rebuild index if stale
        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        # Tokenize the query
        query_tokens = self._tokenizer(query)

        if not query_tokens:
            return []

        # Score each document against the query, keep non-zero scores
        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens=query_tokens, doc_index=i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        # Sort by score (highest first) and return top k results
        raw_scores.sort(key=lambda item: item[0], reverse=True)

        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])
        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)
