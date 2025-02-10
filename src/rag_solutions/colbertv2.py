import json
from typing import Any, Dict, List

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic.v1 import root_validator

from config import COLBERT_MODEL_PATH, CONSOLE, DATA_ROOT_DIR


def run_colbertv2_index(
    dataset_name: str,
    index_name: str,
    corpus_tsv_path: str,
    checkpoint_path: str = COLBERT_MODEL_PATH,
    overwrite: bool = False,
):
    with Run().context(
        RunConfig(
            nranks=1, experiment="colbert", root=f"{DATA_ROOT_DIR}/{dataset_name}/"
        )
    ):
        config = ColBERTConfig(
            nbits=2,
            root=f"{DATA_ROOT_DIR}/{dataset_name}/colbert",
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=corpus_tsv_path, overwrite=overwrite)
        CONSOLE.log(
            f"[green] Indexing done for dataset {dataset_name}, index {index_name}"
        )


def index_colbertv2():
    corpus_names = ["hotpotqa_1000", "musique_1000", "2wikimultihopqa_1000"]
    for corpus_name in corpus_names:
        dataset_name = corpus_name.split("_")[0]
        with open(
            f"{DATA_ROOT_DIR}/{dataset_name}/{corpus_name.replace('1000', 'corpus')}.json",
            "r",
        ) as json_f:
            corpus = json.load(json_f)
            if corpus_name == "hotpotqa_1000":
                corpus_contents = [
                    key + "\t" + "".join(value) for key, value in corpus.items()
                ]
            elif corpus_name == "musique_1000":
                corpus_contents = [
                    item["title"] + "\t" + item["text"].replace("\n", " ")
                    for item in corpus
                ]
            elif corpus_name == "2wikimultihopqa_1000":
                corpus_contents = [
                    item["title"] + "\t" + item["text"].replace("\n", " ")
                    for item in corpus
                ]
            else:
                raise NotImplementedError(f"Corpus {corpus_name} not implemented")

            corpus_tsv_path = (
                f"{DATA_ROOT_DIR}/{dataset_name}/{corpus_name}_colbertv2_corpus.tsv"
            )
            with open(corpus_tsv_path, "w") as f:
                for pid, p in enumerate(corpus_contents):
                    f.write(f'{pid}\t"{p}"' + "\n")
            CONSOLE.log(
                f"[green] Corpus tsv saved: {corpus_tsv_path}", len(corpus_contents)
            )

            run_colbertv2_index(
                dataset_name,
                corpus_name + "_nbits_2",
                corpus_tsv_path,
                COLBERT_MODEL_PATH,
                overwrite=True,
            )


class Colbertv2Retriever(BaseRetriever):

    root: str
    index_name: str
    tsv_path: str
    top_k: int = 5

    searcher: Searcher
    pid_2_text: Dict

    @root_validator(pre=True)
    @classmethod
    def create_searcher(cls, values: Dict) -> Any:
        """Create a ColBERTv2 searcher."""

        with Run().context(
            RunConfig(nranks=1, experiment="colbert", root=values["root"])
        ):
            config = ColBERTConfig(
                root=values["root"].rstrip("/") + "/colbert",
            )
            values["searcher"] = Searcher(index=values["index_name"], config=config)

        with open(values["tsv_path"], "r") as f:
            values["pid_2_text"] = {}
            for line in f:
                pid, text = line.strip().split("\t", 1)
                values["pid_2_text"][int(pid)] = text

        return values

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        queries = Queries(path=None, data={0: query})
        ranking = self.searcher.search_all(queries, k=self.top_k)

        docs = []

        for pid, _, score in ranking.data[0]:
            docs.append(
                Document(
                    page_content=self.pid_2_text[pid],
                    metadata={"pid": pid, "score": score},
                )
            )
        return docs


COLBERTV2RETRIEVER = None


def create_colbertv2_retriever(dataset_name: str, top_k: int = 5) -> Colbertv2Retriever:
    global COLBERTV2RETRIEVER
    """创建colbertv2的检索器"""
    if COLBERTV2RETRIEVER is None:
        COLBERTV2RETRIEVER = Colbertv2Retriever(
            root=f"{DATA_ROOT_DIR}/{dataset_name}/",
            index_name=f"{dataset_name}_1000_nbits_2",
            tsv_path=f"{DATA_ROOT_DIR}/{dataset_name}/{dataset_name}_1000_colbertv2_corpus.tsv",
            top_k=top_k,
        )
    return COLBERTV2RETRIEVER


if __name__ == "__main__":
    # use colbert-v2 to create indexes
    index_colbertv2()

    # test colbert-v2 retriever
    retriever = create_colbertv2_retriever(dataset_name="musique", top_k=5)
    print(retriever.invoke("who is lion messi"))
