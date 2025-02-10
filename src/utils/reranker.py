# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List, Optional, Sequence, Union

from FlagEmbedding import FlagReranker
from langchain.pydantic_v1 import root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import \
    BaseDocumentCompressor
from langchain_core.documents import Document



class BGEReranker():
    """bge reranker wrap

    Args:
        model_path (str): huggingface model path, default to "BAAI/bge-reranker-v2-m3"
    """

    model_path: str = "BAAI/bge-reranker-v2-m3"
    devices: Optional[Union[str, List[str], List[int]]] = None, # specify devices, such as ["cuda:0"] or ["0"]
    client: FlagReranker

    def __init__(self, model_path, devices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = FlagReranker(
            model_path,
            use_fp16=True,
            devices=devices
        )

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        data = [[query, candidate] for candidate in candidates]

        scores = [self.client.compute_score(d) for d in data]
        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        candidates = [doc.page_content for doc in documents]
        scores = self.rerank(query, candidates)
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = score
        documents = sorted(
            documents, key=lambda x: x.metadata["rerank_score"], reverse=True
        )
        return documents