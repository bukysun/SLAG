# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from langchain.chains.base import Chain
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

from luluai.langchain_contrib.retrievers.rerankers import LuLuReranker

from src.prompt_template import (
    ENTITY_LINK_REVIEW_PROMPT,
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    ENTITY_LINK_REVIEW_PROMPT
)

from src.utils.utils import process_entity_with_etf, split_string_by_multi_markers


class EntityLinkMultiSourceChain(Chain):
    """Entity link

    Args:
        vector_store: Vector store for embedding retriving
        graph: Graph store for fulltext index search
        whether_use_reranker: Whether to use reranker
        reranker: reranker for reranking results of vectorstore similarity search
        prerank_topk: number of candidates to be retrieved from vectorstore
        search_timeout: Timeout for embedding search
        search_max_retry: Retry times for search
        input_key: Input key
        output_key: Output key
        return_intermediate_steps: Whether to return intermediate steps.
        intermediate_steps_key: Key for intermediate steps.
    """
    vector_store: VectorStore
    graph: Neo4jGraph
    whether_use_reranker: bool = False
    reranker: Optional[LuLuReranker] = None
    prerank_topk: int = 100
    rerank_topk: int = 1
    search_timeout: float = 2000
    search_max_retry: int = 3
    input_key: str = "filtered_entities"
    output_key: str = "linked_entities"
    question_key: str = "question"
    rewritten_question_key: str = "rewritten_question"
    rewritten_mode: str = "afterfix"     # support two modes of rewritten: afterfix and replace
    return_intermediate_steps: bool = True
    intermediate_steps_key: str = "intermediate_steps"
    whether_rewrite_question: bool = True
    restricted_type: str = ""   # now only support for one type. Todo: support for multiple types
    whether_use_llm_review: bool = True
    review_chain: Optional[RunnableSerializable[Dict, Any]] = None

    def __init__(
        self,
        review_model_name: str,
        whether_use_llm_review: bool = True,
        **kwargs
    ):
        review_chain = None
        if whether_use_llm_review:
            llm = ChatOpenAI(model_name=review_model_name, temperature=0) 
            review_chain = ENTITY_LINK_REVIEW_PROMPT | llm | StrOutputParser()
        
        super().__init__(
            whether_use_llm_review = whether_use_llm_review,
            review_chain = review_chain,
            **kwargs
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys"""
        return [self.input_key, self.question_key]
    

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys"""
        if self.whether_rewrite_question:
            return [self.output_key, self.rewritten_question_key]
        else:
            return [self.output_key]
        

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # search most relevant entities in kg
        entities = list(set(inputs[self.input_key]))
        el_dict = {}
        for ent in entities:
            el_ents = self._search_with_retry(ent, _run_manager)
            if len(el_ents) > 0:
                el_dict[ent] = el_ents

        # review the results by llm
        if el_dict and self.whether_use_llm_review:
            el_dict = self._review_entities(el_dict, inputs[self.question_key], _run_manager)

        new_question = ""
        if self.whether_rewrite_question:
            new_question = self._rewrite_question(el_dict, inputs[self.question_key])
        self._log_it(_run_manager, el_dict, new_question)
        return self._prepare_chain_result(inputs, el_dict, new_question)
    

    def _log_it(self, run_manager, el_dict, new_question):
        run_manager.on_text("EL Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(el_dict), color="green", end="\n", verbose=self.verbose)
        if self.whether_rewrite_question:
            run_manager.on_text("Rewrite question:", end='\n', verbose=self.verbose)
            run_manager.on_text(new_question, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(
            self,
            inputs: Dict[str, Any],
            el_dict: Dict[str, Any],
            rewritten_question: str
    ) -> Dict[str, Any]:
        """Add output and intermediate record"""
        chain_result = {self.output_key: el_dict, self.input_key: inputs[self.input_key],
                        self.question_key: inputs[self.question_key]}
        if self.whether_rewrite_question:
            chain_result[self.rewritten_question_key] = rewritten_question

        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps += [
                {self.output_key: el_dict}
            ]
            if self.whether_rewrite_question:
                intermediate_steps += [
                    {self.rewritten_question_key: rewritten_question}
                ]
            chain_result[self.intermediate_steps_key] = intermediate_steps

        return chain_result
    

    def _search_with_retry(
            self,
            entity: str,
            run_manager: CallbackManagerForChainRun
    ) -> List[Tuple[str, str, str]]:
        """search with multple retry"""
        linked_entities = []
        for i in range(self.search_max_retry):
            try:
                linked_entities = self._search_fuzzy(entity=entity)
            except Exception as e:
                run_manager.on_text(f'try {i} times, link entity {entity} casue error:',
                                    end='\n', verbose=self.verbose)
                run_manager.on_text(f'{e}', color='red', end='\n', verbose=self.verbose)
                continue
            break
        return linked_entities
    

    def _search_fuzzy(self, entity: str) -> List[Tuple[str, str, str]]:
        """fuzzy search relavant entities with embedding and reranker"""
        # query vector store
        if self.restricted_type == "":
            results = self.vector_store.similarity_search(
                entity,
                k=self.prerank_topk,
                param={"ef": 3 * self.prerank_topk + 10},
                timeout=self.search_timeout
            )
        else:
            results = self.vector_store.similarity_search(
                entity,
                k=self.prerank_topk,
                param={"ef": 3 * self.prerank_topk + 10},
                timeout=self.search_timeout,
                expr=f"type == \"{self.restricted_type}\""
            )

        # rerank with bge reranker
        if self.whether_use_reranker:
            results = self.reranker.compress_documents(results, entity)
    
        # parse result
        linked_entities = []
        for r in results:
            types, properties = r.metadata["type"].split('@#@'), r.metadata["property"].split('@#@')
            type_prop = list(zip(types, properties))
            for t, p in type_prop:
                linked_entities.append((t, p, r.page_content))
        return linked_entities
    

    def _review_entities(self, el_dict: Dict[str, List], context: str, run_manager: CallbackManagerForChainRun) -> Dict[str, List]:
        """review entity links with llm"""
        checked_entity_pairs = []
        for ent, linked_list in el_dict.items():
            checked_entity_pairs.extend([f"({ent},{l[2]})" for l in linked_list if l[2] != ent])    # exact matched entities don't need to be review
        
        if len(checked_entity_pairs) > 0:
            review_withcallback = self.review_chain.with_config(callbacks=run_manager.get_child())
            raw_res = review_withcallback.invoke(
                {
                    "context": context,
                    "entity_pairs": ", ".join(checked_entity_pairs)
                }
            )
            parsed_res = parse_review_response(raw_res)

            if not parsed_res:    # no review result, don't change el_dict
                return el_dict
            
            # filter wrong linked entity pairs
            new_el_dict = {}
            for ent, linked_list in el_dict.items():
                tmp = []
                for l in linked_list:
                    linked_ent = l[2]
                    if parsed_res.get((ent, linked_ent), ("true",))[0] == "true":
                        tmp.append(l)
                if tmp:
                    new_el_dict[ent] = tmp
            return new_el_dict
        return el_dict

    def _rewrite_question(self, el_dict: Dict[str, Any], question: str):
        if self.rewritten_mode == "afterfix":
            return self._rewrite_in_afterfix(el_dict, question)
        elif self.rewritten_mode == "replace":
            return self._rewrite_in_replace(el_dict, question)
        else:
            raise ValueError("Unsupported rewritten mode {}".format(self.rewritten_mode))

    def _rewrite_in_afterfix(self, el_dict: Dict[str, Any], question: str):
        """rewrite original question with linked entities

        Args:
            el_dict (Dict[str, Any]): entity in quesion --> (node type, node property, entity name in kg)
            question (str): original question
        Return:
            str: rewritten question
        """
        # format extra information
        el_descriptions = []
        for ent_in_question, linked_entities in el_dict.items():
            if len(linked_entities) == 1:
                node_type, node_property, ent_in_kg = linked_entities[0]
                desc = f"`{ent_in_question}` denotes a `{node_type}` of which the `{node_property}` is `{ent_in_kg}`"
                el_descriptions.append(desc)
            else:
                prefix = f"`{ent_in_question}` may denote "
                descs = []
                for i, (node_type, node_property, ent_in_kg) in enumerate(linked_entities):
                    desc = f"a `{node_type}` of which the `{node_property}` is `{ent_in_kg}`"
                    descs.append(desc)
                el_descriptions.append(prefix + " or ".join(descs))

        # rewrite question by add extra info
        if len(el_descriptions) > 0:
            extra_info_str = ';'.join(el_descriptions)
            new_question = question.strip()
            if not new_question.endswith('?'):
                new_question = new_question + '?'
            new_question = new_question + 'Additional information: ' + extra_info_str
            return new_question
        else:
            return question
        
    def _rewrite_in_replace(self, el_dict: Dict[str, Any], question: str):
        """rewrite original question with replaced linked entities

        Args:
            el_dict (Dict[str, Any]): entity in quesion --> (node type, node property, entity name in kg)
            question (str): original question
        Return:
            str: rewritten question
        """
        question_set = [question]
        for ent_in_question, linked_entities in el_dict.items():
            new_question_set = []
            for _, _, ent_in_kg in linked_entities:
                new_question_set.extend([q.replace(ent_in_question, ent_in_kg) for q in question_set])
            question_set = list(set(new_question_set))

        return "|".join(question_set)



def parse_review_response(res: str):
    """parse result of review agent"""
    records = split_string_by_multi_markers(
            res,
            [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER],
        )

    parsed_rec = defaultdict(dict)
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [DEFAULT_TUPLE_DELIMITER]
        )
        
        if len(record_attributes) == 5:
            source_entity, target_entity, is_same, reason, score = record_attributes
            parsed_rec[(source_entity, target_entity)] = (is_same, reason, score)
    return parsed_rec