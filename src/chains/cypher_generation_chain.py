# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chains.graph_qa.cypher import construct_schema, extract_cypher

from utils.utils import fill_prompt_template


class CypherGenerationChain(Chain):
    """Convert question to cypher query based on LLM.

    Args:
        cypher_gen_chain: LLMChain to generate cypher query.
        graph_schema: Graph schema text.
        predicate_desc_text: Predicate description text.
        return_intermediate_steps: Whether to return intermediate steps.
        input_key: Input key.
        output_key: Output key.
        intermediate_steps_key: Intermediate steps key.
    """
    cypher_gen_chain: RunnableSerializable[Dict, Any]
    graph_schema: str
    predicate_desc_text: str
    return_intermediate_steps: bool
    input_key: str = "rewritten_question"
    output_key: str = "cypher_query"
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
            self,
            llm: BaseLanguageModel,
            prompt_template: BasePromptTemplate,
            graph_structured_schema: Dict[str, Any],
            predicate_desc: List[Dict[str, str]] = [],
            return_intermediate_steps: bool = True,
            exclude_types: List[str] = [],  # exluded node type in graph
            include_types: List[str] = [],  # included node type in graph
    ):
        cypher_gen_chain = prompt_template | llm | StrOutputParser()

        if exclude_types and include_types:
            raise ValueError("`exclude_types` and `include_types` cannot be both set")
        graph_schema = construct_schema(graph_structured_schema, include_types, exclude_types)
        predicate_desc_text = construct_predicate_desc_text(predicate_desc)
        super().__init__(
            cypher_gen_chain=cypher_gen_chain,
            graph_schema=graph_schema,
            predicate_desc_text=predicate_desc_text,
            return_intermediate_steps=return_intermediate_steps,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys"""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys"""
        return [self.output_key]
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Call function"""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        generated_cypher = self._generate_cypher(inputs, _run_manager)
        return self._prepare_chain_result(inputs, generated_cypher)

    def _generate_cypher(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun) -> str:
        """Generate cypher"""
        prepared_inputs = self._prepare_chain_input(inputs)
        cypher_gen_withcallback = self.cypher_gen_chain.with_config(callbacks=run_manager.get_child())
        generated_cypher = cypher_gen_withcallback.invoke(
            input=prepared_inputs
        )
        generated_cypher = extract_cypher(generated_cypher)
        if generated_cypher.startswith("cypher"):  # process output format "cypher:..."
            generated_cypher = generated_cypher[6:].strip()
        self._log_it(generated_cypher, run_manager)
        return generated_cypher

    def _log_it(self, generated_cypher: str, run_manager: CallbackManagerForChainRun):
        run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_input(self, inputs: Dict[str, Any]):
        """Construct slots dict in prompt template"""
        return {
            "question": inputs[self.input_key],
            "graph_schema": self.graph_schema,
            "predicate_description": self.predicate_desc_text
        }

    def _prepare_chain_result(self, inputs: Dict[str, Any], generated_cypher: str) -> Dict[str, Any]:
        """Add output and intermediate record"""
        chain_result = {self.output_key: generated_cypher}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            filled_prompt = fill_prompt_template(self.cypher_gen_chain, self._prepare_chain_input(inputs))
            intermediate_steps += [
                {self.input_key: inputs[self.input_key]},
                {self.output_key: generated_cypher},
                {f"{self.__class__.__name__}_filled_prompt": filled_prompt}
            ]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result



def construct_predicate_desc_text(predicate_desc: List[Dict[str, str]]) -> str:
    """Format predicate description dict to text 

    Args:
        predicate_desc (List[Dict[str, str]]): dict map predicate name to predicate description

    Returns:
        str: predicate description text
    """
    if len(predicate_desc) == 0:
        return ""

    result = ["The description of common relations in the knowledge graph:"]
    for item in predicate_desc:
        item_as_text = f"({item['subject']})-[{item['predicate']}]->({item['object']}): {item['definition']}"
        result.append(item_as_text)
    return "\n".join(result)