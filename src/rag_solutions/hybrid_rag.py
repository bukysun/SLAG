import json
from operator import itemgetter
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)

from config import (
    HOTPOTQA_WITH_NEW_QUERY_PATH,
    MUSIQUE_WITH_NEW_QUERY_PATH,
    TWOWIKIMULTIHOPQA_WIKI_WITH_NEW_QUERY_PATH,
)
from rag_solutions.base import create_llm, truncate_text
from rag_solutions.colbertv2 import create_colbertv2_retriever

NAIVE_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Please give your answer directly without any additional text. Please make the output as concise as possible, containing only a few words.\nContext: {context} \nQuestion: {question} \nAnswer:"""


@chain
def format_docs(docs: List[Document]):
    return truncate_text("\n\n".join(doc.page_content for doc in docs), 3000)


KG_DOCS_DICT = {}


def init_hybrid_rag(path):
    with open(path, "r") as f:
        data = json.load(f)

    global KG_DOCS_DICT

    if not KG_DOCS_DICT:
        for item in data:
            KG_DOCS_DICT[item["question"]] = item.get("node_expand_info", "")


def create_hybrid_rag_chain(
    dataset_name: Literal["musique", "hotpotqa", "2wikimultihopqa"],
    top_k: int,
    llm_name: str = "gpt-3.5-turbo-1106",
) -> Runnable:
    """
    Create a hybrid rag chain.
    """
    path_dict = {
        "musique": MUSIQUE_WITH_NEW_QUERY_PATH,
        "hotpotqa": HOTPOTQA_WITH_NEW_QUERY_PATH,
        "2wikimultihopqa": TWOWIKIMULTIHOPQA_WIKI_WITH_NEW_QUERY_PATH,
    }
    init_hybrid_rag(path_dict[dataset_name])

    prompt = PromptTemplate.from_template(template=NAIVE_PROMPT)

    retriever = create_colbertv2_retriever(dataset_name=dataset_name, top_k=top_k)

    hybrid_retriever = (
        itemgetter("question")
        | RunnableParallel(
            {
                "vector_docs": retriever,
                "kg_docs": lambda x: [Document(page_content=KG_DOCS_DICT[x])],
            }
        )
        | RunnableLambda(lambda x: x["vector_docs"] + x["kg_docs"])
    )

    qa_chain = RunnablePassthrough().assign(
        docs=hybrid_retriever
    ) | RunnablePassthrough.assign(
        predicted_answer=(
            {
                "context": itemgetter("docs") | format_docs,
                "question": itemgetter("question"),
            }
            | prompt
            | create_llm(model_name=llm_name)
            | StrOutputParser()
        )
    )
    return qa_chain


if __name__ == "__main__":

    chain = create_hybrid_rag_chain(
        dataset_name="hotpotqa", top_k=5, llm_name="gpt35-1106"
    )
    result = chain.invoke(
        {
            "question": "What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?"
        },
    )
    print(result)
