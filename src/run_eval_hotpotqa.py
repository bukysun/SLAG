import json
import time
import traceback
from typing import Callable, Dict, Generator, List, TypedDict

from aim import Run
from langchain_core.documents import Document

from config import CONSOLE, HOTPOTQA_JSON_PATH
from eval.hotpotqa_evaluation import exact_match_score, f1_score
from rag_solutions.hybrid_rag import create_hybrid_rag_chain
from rag_solutions.naive_rag import create_naive_rag_chain
from rag_solutions.slag import create_slag_chain


class Sample(TypedDict):
    id: str
    question: str
    answer: str


class Prediction(TypedDict):
    docs: List[Dict]
    predicted_answer: str
    gold_answer: str
    em: float
    f1: float


def document_to_json(doc: Document) -> Dict:
    """
    Convert a Document object to a JSON dictionary.
    """
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def load_samples() -> Generator[Sample, None, None]:

    with open(HOTPOTQA_JSON_PATH, "r") as f:
        data = json.load(f)

    for item in data:
        yield Sample(
            id=item["_id"],
            question=item["question"],
            answer=item["answer"],
        )


def evalute_single_sample(
    func_solution: Callable,
    sample: Sample,
    llm_name: str,
    top_k: int,
):
    """
    Evaluate a single sample.
    """
    dataset_name = "hotpotqa"

    solution = func_solution(
        dataset_name=dataset_name,
        top_k=top_k,
        llm_name=llm_name,
    )
    prediction: Prediction = solution.invoke({"question": sample["question"]})
    em = exact_match_score(prediction["predicted_answer"], sample["answer"])
    f1, _, _ = f1_score(prediction["predicted_answer"], sample["answer"])
    prediction["em"] = em
    prediction["f1"] = f1
    prediction["gold_answer"] = sample["answer"]
    prediction["docs"] = [document_to_json(doc) for doc in prediction["docs"]]

    return prediction


def evaluate_hotpotqa(
    func_solution: Callable,
    llm_name: str = "gpt-3.5-turbo-1106",
    top_k: int = 2,
    method_name: str = "hybrid_rag",
):
    """
    Evaluate the solution.
    """
    sample_gen = load_samples()

    # 初始化平均值列表
    total_em = []
    total_f1 = []
    predictions = []

    run = Run()
    run["hparams"] = {
        "dataset": "hotpotqa",
        "method": method_name,
        "llm_name": llm_name,
        "recall@n": top_k,
    }

    # 使用生成器sample_gen
    for i, sample in enumerate(sample_gen):
        times = 0
        while True:
            try:
                pred = evalute_single_sample(func_solution, sample, llm_name, top_k)
                predictions.append(pred)
                total_em.append(pred["em"])
                total_f1.append(pred["f1"])

                # 计算当前的平均分
                avg_em = sum(total_em) / len(total_em)
                avg_f1 = sum(total_f1) / len(total_f1)

                run.track(avg_em, name="em", step=i)
                run.track(avg_f1, name="f1", step=i)

                # 使用print在同一行上更新输出
                CONSOLE.log(
                    f"[green] Iteration: {i+1}, em: {avg_em:.4f}, f1: {avg_f1:.4f}",
                    end="",
                )
                break
            except Exception:
                print(traceback.print_exc())
                times += 1
                if times > 10:
                    break
                time.sleep(times * 5)
    return predictions


if __name__ == "__main__":
    func_methods = [
        create_naive_rag_chain,
        create_hybrid_rag_chain,
        create_slag_chain,
    ]
    method_names = [
        "naive_rag[colbertv2]",
        "hybrid_rag[colbertv2]",
        "slag[colbertv2]",
    ]

    for func_method, m_name in zip(func_methods, method_names):
        preds = evaluate_hotpotqa(
            func_solution=func_method,
            method_name=m_name,
            top_k=5,
            llm_name="gpt35-1106",
        )
        with open(f"./outputs/hotpotqa/hotpotqa_{m_name}.json", "w") as f:
            json.dump(preds, f, indent=4)
