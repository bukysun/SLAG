# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from typing import Dict, Any, List, Optional

from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.callbacks import CallbackManagerForChainRun


from utils.utils import fill_prompt_template, load_prompt_examples


class ReasonChain(Chain):
    """Simplify the question according to retrieved sub-graphs

    Args:
        llm_chain: LLM chain to call
        return_intermediate_steps: Whether to return intermediate steps
        input_key (str): Input key
        output_key (str): Output key
        question_key (str): Question key
        intermediate_key (str): Key of intermediate steps
    """
    cot_chain: RunnableSerializable[Dict, Any]
    return_intermediate_steps: bool
    background_key: str = "background"
    input_key: str = "question"
    output_key: str = "followup_ans"
    subjects_key: str = "subjects"
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
            self,
            llm: BaseLanguageModel,
            prompt_template: BasePromptTemplate,
            example_prompt_template: BasePromptTemplate,
            example_file: str,
            return_intermediate_steps: bool = True,
            **kwargs
    ):
        examples = load_prompt_examples(example_file)
        example_for_slots = "\n\n".join(
            [e.to_string() for e in example_prompt_template.batch(examples)]
        )
        prompt_template = prompt_template.partial(examples=example_for_slots)

        cot_chain = prompt_template | llm | JsonOutputParser()
            
        super().__init__(
            cot_chain=cot_chain,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs
        )
    
    @property
    def input_keys(self) -> List[str]:
        """Return the input keys"""
        return [self.input_key, self.background_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys"""
        return [self.output_key, self.subjects_key]

    def _prepare_chain_input(self, inputs: Dict[str, Any]):
        """Construct slots dict in prompt template
        """
        return {
            "question": inputs[self.input_key],
            "background": inputs[self.background_key]
        }

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
        cot_res = self._run_cot_chain(inputs, _run_manager)
        self._log_it(_run_manager, cot_res)
        return self._prepare_chain_result(inputs, cot_res)

    def _run_cot_chain(
            self,
            inputs: Dict[str, Any],
            run_manager: CallbackManagerForChainRun
    ) -> Dict[str, Any]:
        prepared_inputs = self._prepare_chain_input(inputs)
        cot_withcallback = self.cot_chain.with_config(callbacks=run_manager.get_child())
        result = cot_withcallback.invoke(input=prepared_inputs)
        return result

    def _log_it(self, run_manager: CallbackManagerForChainRun, cot_res: Dict[str, Any]):
        run_manager.on_text("subjects: {}".format(cot_res["subjects"]), end="\n", verbose=self.verbose)
        run_manager.on_text("followup answer: {}".format(cot_res["answer"]), color="green", end="\n",
                            verbose=self.verbose)

    def _prepare_chain_result(self, inputs: Dict[str, Any], cot_res: Dict[str, Any]):
        """Add output and intermediate record"""
        chain_result = {self.output_key: cot_res["answer"], self.subjects_key: cot_res["subjects"],
                        self.input_key: inputs[self.input_key]}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            filled_prompt = fill_prompt_template(self.cot_chain, self._prepare_chain_input(inputs))
            intermediate_steps += [
                {f"{self.__class__.__name__}_filled_prompt": filled_prompt},
                {self.output_key: cot_res["answer"]},
                {self.subjects_key: cot_res["subjects"]}
            ]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result