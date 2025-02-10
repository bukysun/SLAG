import json
from typing import Any, Dict, List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, chain
from sklearn.metrics.pairwise import cosine_similarity
from rag_solutions.base import create_cached_embedding


class SampleSelector:
    """
    A sample selector that selects the most similar sample to the query.
    """

    samples: List[Dict] = []
    candidate_vectors: List[List[float]] = []
    embedding: Embeddings = create_cached_embedding()
    sample_path: str = None

    @classmethod
    def set_sample_path(cls, path: str) -> None:
        """设置样本路径"""
        if cls.sample_path != path:
            cls.sample_path = path
            cls.samples = []
        cls.set_samples()

    @classmethod
    def set_samples(cls) -> None:
        """设置样本"""
        if not cls.sample_path:
            raise ValueError("sample_path is not set")
        if not cls.samples:
            cls.samples = parse_prompt(cls.sample_path)
            cls.candidate_vectors = [
                cls.embedding.embed_query(sample["question"]) for sample in cls.samples
            ]

    @classmethod
    def select_samples(cls, query: str, top_k: int = 1) -> List[Dict]:
        """选择样本"""
        query_vector = cls.embedding.embed_query(query)
        scores = cosine_similarity([query_vector], cls.candidate_vectors)
        scores = scores[0]
        return [cls.samples[i] for i in np.argsort(scores)[::-1][:top_k]]


def parse_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split("# METADATA: ")
    parsed_data = []

    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split("\n", 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split("\n\nQ: ")
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split("\nA: ")
        question = qa_pair[0].strip()
        thought_and_answer = qa_pair[1].strip().split("So the answer is: ")
        thought = thought_and_answer[0].strip()
        answer = thought_and_answer[1].strip()

        parsed_data.append(
            {
                "metadata": metadata,
                "document": document_text,
                "question": question,
                "thought_and_answer": qa_pair[1].strip(),
                "thought": thought,
                "answer": answer,
            }
        )

    return parsed_data


cot_system_instruction = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response in a few words, devoid of additional elaborations.'
)


def cot_prompt(few_shot_path: str) -> Runnable:
    SampleSelector.set_sample_path(few_shot_path)

    @chain
    def wrapped_cot_prompt(inputs: Dict[str, Any]) -> List[BaseMessage]:
        question = inputs["question"]
        context = inputs["context"]
        return construct_cot_prompt(question, context)

    return wrapped_cot_prompt


def construct_cot_prompt(question: str, context: str, num: int = 3) -> str:
    instruction = cot_system_instruction
    messages = [SystemMessage(instruction)]
    few_shot = SampleSelector.select_samples(query=question, top_k=num)
    for sample in few_shot:
        if "document" in sample:  # document and question from user
            cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}'
        else:  # no document, only question from user
            cur_sample = f'Question: {sample["question"]}'
        if "thought" in sample:  # Chain-of-Thought
            messages.append(HumanMessage(cur_sample + "\nThought: "))
            messages.append(
                AIMessage(f'{sample["thought"]}\nAnswer: {sample["answer"]}')
            )
        else:  # No Chain-of-Thought, directly answer the question
            messages.append(HumanMessage(cur_sample + "\nAnswer: "))
            messages.append(AIMessage(f'Answer: {sample["answer"]}'))

    user_prompt = f"{context}\n\nQuestion: {question}\nThought: "
    messages.append(HumanMessage(user_prompt))

    if few_shot:
        assert len(messages) == len(few_shot) * 2 + 2
    else:
        assert len(messages) == 2
    # messages = ChatPromptTemplate.from_messages(messages).format_prompt().to_messages()
    return messages
