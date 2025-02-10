import json
from operator import itemgetter
from typing import Callable, List, Literal

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
    HOTPOTQA_FEW_SHOT_PATH,
    HOTPOTQA_WITH_NEW_QUERY_PATH,
    MUSIQUE_FEW_SHOT_PATH,
    MUSIQUE_WITH_NEW_QUERY_PATH,
    TWOWIKIMULTIHOPQA_FEW_SHOT_PATH,
    TWOWIKIMULTIHOPQA_WIKI_WITH_NEW_QUERY_PATH,
)
from rag_solutions.base import create_llm
from rag_solutions.colbertv2 import create_colbertv2_retriever
from rag_solutions.prompt import cot_prompt

NAIVE_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Please give your answer directly without any additional text. Please make the output as concise as possible, containing only a few words.\nContext: \n{context} \nQuestion: {question} \nAnswer:"""


@chain
def format_docs(docs: List[Document]):
    contents = []

    for doc in docs:
        content = doc.page_content.strip().strip('"')
        if "\t" in doc.page_content:
            content = "Wikipedia Title: " + "\n".join(content.split("\t", 1))
        else:
            content = "Infomation from the Knowledge Graph: " + content
        contents.append(content)
    return "\n\n".join(contents)


KG_DOCS_DICT = {}


def init_slag(path):
    with open(path, "r") as f:
        data = json.load(f)
    global KG_DOCS_DICT

    if not KG_DOCS_DICT:
        for item in data:
            KG_DOCS_DICT[item["question"]] = item.get("new_question", None)


def create_slag_retriever(
    dataset_name: Literal["musique", "hotpotqa", "2wikimultihopqa"],
    top_k: int,
) -> Runnable:
    retriever = create_colbertv2_retriever(dataset_name=dataset_name, top_k=top_k)

    @chain
    def retrieve(question: str) -> List[Document]:
        kg_answer = KG_DOCS_DICT.get(question, "")
        docs = []
        if kg_answer is None:
            return docs
        else:
            if "New question:" in kg_answer:
                info = kg_answer.split("New question:")
                context, new_question = info[0].strip(), info[1].strip()
                docs.append(
                    Document(page_content=context, metadata={"new_question": kg_answer})
                )
            else:
                docs.append(
                    Document(
                        page_content=kg_answer, metadata={"new_question": kg_answer}
                    )
                )
                new_question = question
            docs.extend(retriever.invoke(new_question))
        return docs

    return retrieve


def create_slag_chain(
    dataset_name: Literal["musique", "hotpotqa", "2wikimultihopqa"],
    top_k: int,
    llm_name: str = "gpt-3.5-turbo-1106",
    is_naive_prompt: bool = False,
):
    """
    Create a hybrid rag chain.
    """

    if is_naive_prompt:
        prompt = PromptTemplate.from_template(template=NAIVE_PROMPT)
    else:
        if dataset_name == "musique":
            prompt = cot_prompt(MUSIQUE_FEW_SHOT_PATH)
        elif dataset_name == "2wikimultihopqa":
            prompt = cot_prompt(TWOWIKIMULTIHOPQA_FEW_SHOT_PATH)
        elif dataset_name == "hotpotqa":
            prompt = cot_prompt(HOTPOTQA_FEW_SHOT_PATH)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    path_dict = {
        "musique": MUSIQUE_WITH_NEW_QUERY_PATH,
        "hotpotqa": HOTPOTQA_WITH_NEW_QUERY_PATH,
        "2wikimultihopqa": TWOWIKIMULTIHOPQA_WIKI_WITH_NEW_QUERY_PATH,
    }
    init_slag(path_dict[dataset_name])

    retriever = create_slag_retriever(dataset_name=dataset_name, top_k=top_k)

    qa_chain = RunnablePassthrough().assign(
        docs=itemgetter("question") | retriever
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

    if not is_naive_prompt:
        qa_chain = qa_chain | RunnablePassthrough.assign(
            predicted_answer=itemgetter("predicted_answer")
            | RunnableLambda(lambda x: x.split("Answer:")[-1].strip()),
            llm_output=itemgetter("predicted_answer"),
        )

    return qa_chain


if __name__ == "__main__":
    chain = create_slag_chain(
        dataset_name="musique",
        top_k=5,
        llm_name="gpt35-1106",
    )
    result = chain.invoke(
        {
            "question": "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
        },
    )
    print(result)
