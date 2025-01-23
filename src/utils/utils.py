# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/09/19 15:39:10
# @Author  : wuhui
# @Email   : wuhui@efunds.com.cn
# @File    : utils.py

import os
import re
import json
import asyncio
from urllib import parse
from typing import Dict, Any, Optional, List

from pymilvus.exceptions import ConnectionConfigException
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_openai.chat_models import ChatOpenAI

from luluai.utils.ai_model_sdk.constants import DEV_BASE_URL
from luluai.langchain_contrib.retrievers.rerankers import LuLuReranker



def fill_prompt_template(llm_chain: RunnableSerializable[Dict, Any], inputs: Dict[str, Any]) -> str:
    """
    Fill the prompt template with the given inputs.
    """
    return llm_chain.get_prompts()[0].format_prompt(**inputs).to_string()


def parse_milvus_conn_args(uri: Optional[str] = None):
    """parse connection args from uri or environment

    Args:
        uri (_type_, optional): _description_. Defaults to None.
    """
    if uri is None:
        uri = os.environ.get("MILVUS_URI")
    assert uri is not None, "Argument uri must be set or environment variable MILVUS_URI must be set"

    illegal_uri_msg = (
        "Illegal uri: [{}], expected form 'http[s]://[user:password@]example.com[:12345]'"
    )

    try:
        parsed_uri = parse.urlparse(uri)
    except Exception as e:
        raise ConnectionConfigException(
            message=f"{illegal_uri_msg.format(uri)}: <{type(e).__name__}, {e}>"
        ) from e

    conn = {}
    conn["uri"] = uri
    conn["host"] = parsed_uri.hostname
    conn["port"] = parsed_uri.port
    conn["user"] = parsed_uri.username
    conn["password"] = parsed_uri.password
    return conn


def load_chat_model(model_name: str) -> BaseLanguageModel:
    """load chat model"""
    return ChatOpenAI(model=model_name, temperature=0)


def load_prompt_examples(file_name: str) -> List[Dict[str, str]]:
    """load batch demonstrations from file for prompt"""
    examples = []
    with open(file_name, 'r', encoding='utf-8') as fr:
        for l in fr.readlines():
            record = json.loads(l.strip())
            for k, v in record.items():
                if not isinstance(v, str):
                    v = str(v)
                    record[k] = v
            examples.append(record)
    return examples


def create_lulu_reranker():
    return LuLuReranker(
        aimodel_appid="AIGC",
        aimodel_secretkey="feyji5b7s0crls3pdsyg78g3",
        aimodel_environment_url=DEV_BASE_URL
    )


def single2double_quote(ai_message: AIMessage):
    """replace single quote with double quote in the response of LLM
    """
    text = ai_message.content
    ai_message.content = text.replace("'", "\"")
    return ai_message

def string_replace(input_str: str, a:str, b:str):
    """Replace a in input_str with b"""
    return input_str.replace(a, b)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print("Creating a new event loop in main thread.")
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()
    return loop



def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)

def write_json(json_obj, file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)



def graph_result_contains_triple(graph_result_entry: Dict[str, Any]) -> bool:
    """Whether the graph result contains triple"""
    return len(get_triples_from_graph_result(graph_result_entry)) > 0


def get_triples_from_graph_result(graph_result_entry: Dict[str, Any]) -> List[dict]:
    """Return all the triples from graph result
    """
    return [value for key, value in graph_result_entry.items() if isinstance(value, tuple)]


def get_node_name_in_kg(node: Dict[str, Any]) -> Optional[str]:
    """Get name property for a node"""
    if 'name' in node:
        return node['name']
    else:
        return None
