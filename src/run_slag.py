# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import argparse
from argparse import Namespace
from dotenv import load_dotenv
from typing import TypedDict

from langchain.chains.base import Chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_milvus import Milvus


from prompt_template import (
    RESTRICT_NER_PROMPT,
    RESTRICT_NER_EXAMPLE_PROMPT,
    REASON_PROMPT,
    REASON_EXAMPLE_PROMPT
)
from chains.slag_chain import SLAGChainConfig, SLAGChain
from utils.utils import parse_milvus_conn_args
from utils.reranker import BGEReranker


def get_args():
    parser = argparse.ArgumentParser("slag")
    parser.add_argument("--ner_llm_name", type=str, default="gpt-4o", help="llm model for ner task")
    parser.add_argument("--reason_llm_name", type=str, default="gpt-4o", help="llm model for reason task")
    parser.add_argument("--use_llm_review", action="store_true", help="use llm review in entity link")
    parser.add_argument("--review_llm_name", type=str, default="gpt-4o", help="llm model for llm review")
    parser.add_argument("--dataset_name", type=str, choices=["2wikimultihopqa", "hotpotqa", "musique"], help="name of the dataset")
    parser.add_argument("--reranker_model_path", type=str, default="BAAI/bge-reranker-v2-m3", help="model path of BGE reranker, can be set to remote huggingface path")
    parser.add_argument("--run_mode", type=str, choices=["batch", "server"], required=True, help="run mode, batch for batch running, server for single test")
    return parser.parse_args()

def build_slag_chain(args: Namespace):
    """Build slag chain
    """
    graph = Neo4jGraph()
    
    # create vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    milvus_conn_args = parse_milvus_conn_args()
    collection_name = f"public_rag_dataset_{args.dataset_name}_node_index_ada"
    vector_store = Milvus(
        embedding_model, collection_name=collection_name, connection_args=milvus_conn_args
    )

    config = SLAGChainConfig.model_construct(
        graph=graph,
        vector_store=vector_store,
        ner_llm=ChatOpenAI(model=args.ner_llm_name, temperature=0.0),
        ner_prompt=RESTRICT_NER_PROMPT,
        ner_example_prompt=RESTRICT_NER_EXAMPLE_PROMPT,
        ner_example_file="demonstrations/ner_examples.json",
        review_model_name=args.review_llm_name,
        whether_use_llm_review = args.use_llm_review,
        rerank_topk = 10 if args.use_llm_review else 1,
        reranker=BGEReranker(model_path=args.reranker_model_path, devices="cuda:0"),
        reason_llm=ChatOpenAI(model=args.reason_llm_name, temperature=0.0),
        reason_prompt=REASON_PROMPT,
        reason_example_prompt=REASON_EXAMPLE_PROMPT,
        reason_example_file=f"demonstrations/reason_examples.json",
        return_intermediate_steps=False
    )

    chain = SLAGChain(
        config=config
    )

    return chain


def batch_run(args: Namespace, chain: Chain):
    dataset_name = args.dataset_name
    with open(f"./data/{dataset_name}/{dataset_name}.json", "r") as fr:
        dataset = json.load(fr)
    
    reserved_keys = ["named_entities", "linked_entities", "node_expand_info", "subjects"]
    for i, d in enumerate(dataset[:2]):
        print(i)
        res = chain.invoke({"question": d["question"]})
        for k in reserved_keys:
            d[k] = res[chain.output_key][k]
        d["new_question"] = res[chain.output_key]["followup_ans"]

    with open(f"./outputs/{dataset_name}/{dataset_name}_with_newquery_dataset.json", "w") as fw:
        json.dump(dataset, fw, indent=4, ensure_ascii=False)

# create chains
class ChainInput(TypedDict):
    question: str


def run_server(args: Namespace, chain: Chain):
    import uvicorn
    from fastapi import FastAPI
    from langserve import add_routes
    
    app = FastAPI(
        title = "SLAG chain single test",
        version = 1.0,
        description= "A api server for test SLAG chain"
    )

    add_routes(
        app,
        chain.with_types(input_type=ChainInput),
        path = "/slag"
    )
    uvicorn.run(app, host="0.0.0.0", port=7105)


if __name__ == "__main__":
    load_dotenv(override=True)
    args = get_args()
    slag_chain = build_slag_chain(args)    

    if args.run_mode == "batch":
        batch_run(args, slag_chain)
    elif args.run_mode == "server":
        run_server(args, slag_chain)
    else:
        raise ValueError("Unexpected run mode {}, only support batch, server.".format(args.run_mode))
    








