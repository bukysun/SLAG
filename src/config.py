from rich import console

CONSOLE = console.Console()

LLM_CACHE_DB_PATH = "./static/llm_cache_4_slag.db"
EMBEDDING_CACHE_FILE_PATH = "./static/embedding_cache_4_slag/"

# data path
MUSIQUE_JSON_PATH = "./data/musique/musique.json"
MUSIQUE_WITH_NEW_QUERY_PATH = "./outputs/musique/musique_with_newquery_dataset.json"
MUSIQUE_FEW_SHOT_PATH = (
    "./data/musique/gold_with_3_distractors_context_cot_qa_codex.txt"
)

HOTPOTQA_JSON_PATH = "./data/hotpotqa/hotpotqa.json"
HOTPOTQA_WITH_NEW_QUERY_PATH = "./outputs/hotpotqa/hotpotqa_with_newquery_dataset.json"
HOTPOTQA_FEW_SHOT_PATH = (
    "./data/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt"
)

TWOWIKIMULTIHOPQA_WIKI_JSON_PATH = "./data/2wikimultihopqa/2wikimultihopqa.json"
TWOWIKIMULTIHOPQA_WIKI_WITH_NEW_QUERY_PATH = (
    "./outputs/2wikimultihopqa/2wikimultihopqa_with_newquery_dataset.json"
)
TWOWIKIMULTIHOPQA_FEW_SHOT_PATH = (
    "./data/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt"
)

# model path
COLBERT_MODEL_PATH = "./models/colbertv2.0"

DATA_ROOT_DIR = "./data"
OUTPUT_ROOT_DIR = "./outputs"
