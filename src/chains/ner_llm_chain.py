# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import JsonOutputParser

from src.utils.utils import fill_prompt_template, load_prompt_examples, single2double_quote


class NERLLMChain(Chain):
    """Named entity recognition chain with llm.

    Args:
        ner_chain: LLMChain for detecting named entities
        return_intermediate_steps: Whether to return intermediate steps.
        input_key: Input key.
        output_key: Output key.
        intermediate_steps_key: Key for intermediate steps.
    """
    ner_chain: RunnableSerializable[Dict, Any]
    return_intermediate_steps: bool
    input_key: str = "question"
    output_key: str = "named_entities"
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
            self,
            llm: BaseLanguageModel,
            prompt_template: BasePromptTemplate,
            example_prompt_template: Optional[BasePromptTemplate] = None,
            example_file: str = "",
            return_intermediate_steps: bool = True,
            **kwargs
    ):
        if example_prompt_template:
            examples = load_prompt_examples(example_file)
            example_for_slots = "\n\n".join(
                [e.to_string() for e in example_prompt_template.batch(examples)]
            )
            prompt_template = prompt_template.partial(examples=example_for_slots)

        ner_chain = prompt_template | llm | single2double_quote | JsonOutputParser()
        super().__init__(
            ner_chain=ner_chain,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys"""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys"""
        return [self.output_key]

    def _prepare_chain_input(self, inputs: Dict[str, Any]):
        """Construct slots dict in prompt template
        """
        return {
            "question": inputs[self.input_key]
        }

    def _detect_entities(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun) -> List[str]:
        """detect named entities

        Args:
            inputs (Dict[str, Any]): Input dict
            run_manager (CallbackManagerForChainRun): Call back manager

        Returns:
            List[str]: List of detected entities
        """
        prepared_inputs = self._prepare_chain_input(inputs)
        ner_withcallback = self.ner_chain.with_config(callbacks=run_manager.get_child())
        result = ner_withcallback.invoke(input=prepared_inputs)
        detected_entities = result[self.output_key]
        return detected_entities

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Call the chain and return the result.

        Args:
            inputs (Dict[str, Any]): Input dict
            run_manager (CallbackManagerForChainRun): Call back manager

        Returns:
            Dict[str, Any]: Output dict
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        ner_result = self._detect_entities(inputs, _run_manager)
        self._log_it(_run_manager, ner_result)
        return self._prepare_chain_result(inputs, ner_result)

    def _prepare_chain_result(self, inputs: Dict[str, Any], ner_result: List[str]) -> Dict[str, Any]:
        """Add output and intermediate record"""
        chain_result = {self.output_key: ner_result, self.input_key: inputs[self.input_key]}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            filled_prompt = fill_prompt_template(self.ner_chain, self._prepare_chain_input(inputs))
            intermediate_steps += [
                {self.output_key: ner_result},
                {f"{self.__class__.__name__}_filled_prompt": filled_prompt}
            ]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result

    def _log_it(self, run_manager: CallbackManagerForChainRun, entities: List[str]):
        run_manager.on_text("detected entities:", end="\n", verbose=self.verbose)
        run_manager.on_text(','.join(entities), color="green", end="\n", verbose=self.verbose)
