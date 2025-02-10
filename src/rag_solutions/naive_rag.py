from operator import itemgetter
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, chain

from rag_solutions.base import create_llm
from rag_solutions.colbertv2 import create_colbertv2_retriever

NAIVE_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Please give your answer directly without any additional text. Please make the output as concise as possible, containing only a few words.\nContext: {context} \nQuestion: {question} \nAnswer:"""


@chain
def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def create_naive_rag_chain(
    dataset_name: Literal["musique", "hotpotqa", "2wikimultihopqa"],
    top_k: int,
    llm_name: str = "gpt-3.5-turbo-1106",
):
    """
    Create a naive rag chain.
    """
    prompt = PromptTemplate.from_template(template=NAIVE_PROMPT)

    retriever = create_colbertv2_retriever(dataset_name=dataset_name, top_k=top_k)

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
    return qa_chain


if __name__ == "__main__":
    chain = create_naive_rag_chain(
        dataset_name="musique", top_k=2, llm_name="gpt35-1106"
    )
    result = chain.invoke({"question": "who a u?"})
    print(result)
