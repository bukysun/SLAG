# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from operator import itemgetter
from typing import List, Dict, Optional, Any, Tuple

from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain_core.prompts import BasePromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough
)
from langchain_community.graphs import Neo4jGraph

from chains.ner_llm_chain import NERLLMChain
from chains.entity_link_chain import EntityLinkChain
from chains.node_expansion_chain import NodesExpansionChain
from chains.cypher_generation_chain import CypherGenerationChain
from chains.cypher_preprocess_chain import CypherPreprocessorsChain
from chains.cypher_query_chain import CypherQueryChain
from chains.reason_chain import ReasonChain

from utils.tools import CypherPreprocessor
from utils.reranker import BGEReranker


class SLAGChainConfig(BaseModel):
    """Configuration for SLAGChain"""
    class Config:
        arbitrary_types_allowed: bool = True
    
    
    graph: Neo4jGraph
    vector_store: VectorStore

    ner_llm: BaseLanguageModel
    ner_prompt: BasePromptTemplate
    ner_example_prompt: Optional[BasePromptTemplate] = None
    ner_example_file: str = ""
    
    review_model_name: str = "gpt-4o"
    return_intermediate_steps: bool = True
    rerank_topk: int = 10
    whether_rewrite_question: bool = True
    whether_use_reranker: bool = True
    whether_use_llm_review: bool = True
    reranker: BGEReranker

    cypher_gen_llm: BaseLanguageModel
    cypher_gen_prompt: Optional[BasePromptTemplate] = None
    cypher_query_preprocessors: List[CypherPreprocessor] = []
    predicate_descriptions: List[Dict[str, str]] = []
    schema_error_string: str = "SCHEMA_ERROR"

    graph_navi_mode: str = "explore"     # three modes "explore", "exploit", "exploit-explore"     

    reason_llm: BaseLanguageModel
    reason_prompt: BasePromptTemplate
    reason_example_prompt: BasePromptTemplate
    reason_example_file: str = ""

    


class SLAGChain(Chain):
    """The main chain implementing SLAG

    Args:
        pipeline_chain(RunnableSequence): pipeline chain
        return_intermediate_steps(bool): whether to return intermediate steps
        input_key(str): input key
        output_key(str): output key
        intermediate_steps_key(str): intermediate steps key
    """
    pipeline_chain: RunnableSequence
    return_intermediate_steps: bool
    input_key: str = "question"
    output_key: str = "slag_output"
    intermediate_steps_key: str = "intermediate_steps"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def __init__(self, config: SLAGChainConfig):
        pipeline_chain = self._build_chain(config)
        super().__init__(
            pipeline_chain=pipeline_chain,
            return_intermediate_steps=config.return_intermediate_steps
        )

    def _call(self,
              inputs: Dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None
              ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        chain_withcallback = self.pipeline_chain.with_config(callbacks=_run_manager.get_child())
        chain_result = chain_withcallback.invoke(inputs)
        result = {self.output_key: chain_result}
        if self.return_intermediate_steps:
            result[self.intermediate_steps_key] = chain_result[self.intermediate_steps_key]
        return result
    
    def _build_chain(self, config: SLAGChainConfig):
        ner_chain = self._build_ner_chain(config)
        ent_link_chain = self._build_entity_link_chain(config)
        explore_chain = self._build_exploration_chain(config)
        reason_chain = self._build_reason_chain(config)
        graph_navi_chain = explore_chain | RunnablePassthrough.assign(
            background = itemgetter(explore_chain.output_key)
        )
        
        return ner_chain | ent_link_chain | graph_navi_chain | reason_chain
        

    def _build_ner_chain(self, config: SLAGChainConfig) -> NERLLMChain:
        return NERLLMChain(
            llm=config.ner_llm,
            prompt_template=config.ner_prompt,
            return_intermediate_steps=config.return_intermediate_steps,
            example_prompt_template=config.ner_example_prompt,
            example_file=config.ner_example_file
        )

    def _build_entity_link_chain(self, config: SLAGChainConfig) -> EntityLinkChain:
        return EntityLinkChain(
            input_key="named_entities",
            vector_store=config.vector_store,
            graph=config.graph,
            search_timeout=1000,
            whether_rewrite_question=config.whether_rewrite_question,
            whether_use_reranker=config.whether_use_reranker,
            reranker=config.reranker,
            rerank_topk=config.rerank_topk,
            whether_use_llm_review=config.whether_use_llm_review,
            review_model_name=config.review_model_name,
            return_intermediate_steps=config.return_intermediate_steps
        )

    def _build_exploration_chain(self, config: SLAGChainConfig) -> NodesExpansionChain:
        return NodesExpansionChain(
            graph=config.graph,
            max_hop=1,
            min_hop=1,
            limited_record_num=100,
            return_intermediate_steps=config.return_intermediate_steps
        )
    
    def _build_exploition_chain(self, config: SLAGChainConfig) -> RunnableSequence:
        cypher_gen_chain = CypherGenerationChain(
            llm=config.cypher_gen_llm,
            prompt_template=config.cypher_gen_prompt,
            graph_structured_schema=config.graph.get_structured_schema,
            predicate_desc=config.predicate_descriptions,
            return_intermediate_steps=config.return_intermediate_steps
        )

        cypher_prep_chain = CypherPreprocessorsChain(
            cypher_preprocessors=config.cypher_query_preprocessors,
            return_intermediate_steps=config.return_intermediate_steps
        )

        cypher_query_chain = CypherQueryChain(
            graph=config.graph,
            return_intermediate_steps=config.return_intermediate_steps
        )

        return cypher_gen_chain | cypher_prep_chain | cypher_query_chain

    
    def _build_reason_chain(self, config:SLAGChainConfig) -> ReasonChain:
        return ReasonChain(
            llm=config.reason_llm,
            prompt_template=config.reason_prompt,
            example_prompt_template=config.reason_example_prompt,
            example_file=config.reason_example_file
        )
    





    