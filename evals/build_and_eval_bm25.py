#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25 retrieval + recall@k for eval baselines (PRD §8.4 B0).

Vendored under evals/ so CI can run smoke eval without the gitignored
planning/step1-naive-rag/ copy. Keep in sync with that script when changing
scoring logic.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[一-鿿]"


def load_chunks(paths):
    chunks = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


class BM25:
    """Okapi BM25 via numpy + sklearn CountVectorizer (token counts only)."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(
            token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), lowercase=True
        )

    def fit(self, texts):
        self.texts = texts
        X = self.vectorizer.fit_transform(texts)
        self.X = X.tocsr()
        self.doc_len = np.asarray(X.sum(axis=1)).ravel()
        self.avgdl = self.doc_len.mean()
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        n_docs = X.shape[0]
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def search(self, query, k=5):
        q_tokens = self.vectorizer.build_analyzer()(query)
        vocab = self.vectorizer.vocabulary_
        q_term_idx = [vocab[t] for t in q_tokens if t in vocab]
        if not q_term_idx:
            return []
        scores = np.zeros(self.X.shape[0])
        for ti in set(q_term_idx):
            tf = np.asarray(self.X[:, ti].todense()).ravel()
            idf = self.idf[ti]
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
            scores += idf * (tf * (self.k1 + 1)) / np.where(denom == 0, 1, denom)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def hit(chunk_text, keywords):
    return all(kw in chunk_text for kw in keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"载入 {len(chunks)} 个 chunk (来自 {len(args.chunks)} 个文件)")

    bm25 = BM25().fit([c["text"] for c in chunks])
    queries = [json.loads(line) for line in open(args.queries, encoding="utf-8") if line.strip()]

    hits, misses = 0, []
    by_domain = {}
    for q in queries:
        results = bm25.search(q["query"], k=args.k)
        hit_chunks = [chunks[i] for i, _ in results]
        matched = [c for c in hit_chunks if hit(c["text"], q["expect_keywords"])]
        ok = bool(matched)
        hits += ok
        domains_seen = {c.get("domain", "?") for c in hit_chunks}
        for d in domains_seen:
            by_domain.setdefault(d, [0, 0])
            by_domain[d][1] += 1
            if ok:
                by_domain[d][0] += 1
        if not ok:
            misses.append(
                {
                    "query": q["query"],
                    "expect": q["expect_keywords"],
                    "got": [
                        {
                            "chunk_id": chunks[i].get("chunk_id", chunks[i].get("id", "?")),
                            "score": round(s, 3),
                            "domain": chunks[i].get("domain", "?"),
                            "source": chunks[i].get("source_file", chunks[i].get("source", "?")),
                            "head": chunks[i]["text"][:70],
                        }
                        for i, s in results[:3]
                    ],
                }
            )

    n = len(queries)
    print(f"\nrecall@{args.k} = {hits}/{n} = {hits/n:.1%}  (BM25 baseline = PRD §8.4 B0)\n")
    print(f"没命中的 {len(misses)} 条：\n")
    for m in misses:
        print(f"  Q: {m['query']}")
        print(f"     期望包含: {m['expect']}")
        for g in m["got"]:
            print(f"     实际召回: {g['score']} [{g['domain']}] {g['source']} | {g['head']}…")
        print()

    with open("misses.json", "w", encoding="utf-8") as f:
        json.dump(misses, f, ensure_ascii=False, indent=2)
    with open("recall_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "recall_at_k": hits / n,
                "k": args.k,
                "n_queries": n,
                "n_hits": hits,
                "n_chunks": len(chunks),
                "method": "BM25 (B0 baseline per PRD §8.4)",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("失败案例 -> misses.json ，汇总数字 -> recall_summary.json")


if __name__ == "__main__":
    main()
