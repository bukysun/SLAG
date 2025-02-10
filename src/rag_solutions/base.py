from typing import Dict

import tiktoken
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import _llm_cache, set_llm_cache
from langchain.storage import LocalFileStore
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import EMBEDDING_CACHE_FILE_PATH, LLM_CACHE_DB_PATH

load_dotenv()


def create_embedding() -> OpenAIEmbeddings:
    """创建embedding"""
    return OpenAIEmbeddings(model="text-embedding-ada-002")


def create_cached_embedding() -> CacheBackedEmbeddings:
    """创建缓存embedding"""
    underlying_embeddings = create_embedding()
    store = LocalFileStore(EMBEDDING_CACHE_FILE_PATH)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=store,
        namespace=underlying_embeddings.model,
        query_embedding_cache=store,
    )


def set_cache():
    """设置缓存"""
    if _llm_cache is None:
        set_llm_cache(SQLiteCache(database_path=LLM_CACHE_DB_PATH))


def create_llm(model_name: str) -> ChatOpenAI:
    """创建llm"""
    set_cache()

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        cache=True,
    )


def truncate_text(text: str, max_token_length: int = 3000) -> str:
    """截断文本"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_token_length]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text
