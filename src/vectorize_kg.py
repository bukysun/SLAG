# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from dotenv import load_dotenv

from langchain_milvus.vectorstores import Milvus
from langchain_community.graphs import Neo4jGraph
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from utils.utils import parse_milvus_conn_args
from utils.utils import compute_mdhash_id

load_dotenv(override=True)


if __name__ == "__main__":
    load_dotenv(override=True)
    graph = Neo4jGraph()

    dataset_list = ["2wikimultihopqa", "hotpotqa", "musique"]
    node_type_list = ["n_2wikimultihopqa", "hotpotqa", "musique"]
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    milvus_conn_args = parse_milvus_conn_args()

    for i, dataset_name in enumerate(dataset_list):
        collection_name = f'public_rag_dataset_{dataset_name}_node_index_ada'
        vector_store = Milvus(embedding_model, collection_name=collection_name, connection_args=milvus_conn_args)

        # query nodes in graph
        rsp = graph.query(f"MATCH (n:{node_type_list[i]}) RETURN n.name AS name")

        docs, ids = [], []
        for r in rsp:
            content = r["name"]
            doc_id = compute_mdhash_id(content)
            docs.append(Document(page_content=r["name"], metadata={"type": node_type, "property": "name"}))
            ids.append(doc_id)
        
        idx = 0 
        batch_size = 1000
        print(f"Vectorizing knowledge graph for {dataset_name}...")
        while idx < len(docs):
            batch = docs[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            print(i, len(batch))
            vector_store.add_documents(batch, ids=batch_ids)
            i += batch_size
        print(f"Vectorize finished.")
