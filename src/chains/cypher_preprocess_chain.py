# !/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun

from utils.tools import CypherPreprocessor


class CypherPreprocessorsChain(Chain):
    """Preprocess cypher query.

    Args:
        cypher_preprocessors: List of cypher preprocessors.
        input_key: Input key.
        output_key: Output key.
        return_intermediate_steps: Whether to return intermediate steps.
        intermediate_steps_key: Intermediate steps key.
    """
    cypher_preprocessors: List[CypherPreprocessor]
    return_intermediate_steps: bool = True
    input_key: str = "cypher_query"
    output_key: str = "preprocessed_cypher_query"
    intermediate_steps_key: str = "intermediate_steps"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """call function"""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        generated_cypher = inputs[self.input_key]
        preprocessed_cypher, intermediate_steps = self._run_preprocessors(_run_manager, generated_cypher)
        return self._prepare_chain_result(inputs, preprocessed_cypher, intermediate_steps)

    def _run_preprocessors(
            self, _run_manager: CallbackManagerForChainRun, generated_cypher: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Run the preprocessors on the generated cypher query."

        Args:
            _run_manager (CallbackManagerForChainRun): Callback manager for the chain run.
            generated_cypher (str): input cypher

        Returns:
            Tuple[str, List[Dict[str, str]]]: processed cypher query and intermediate steps
        """
        intermediate_steps = []
        for processor in self.cypher_preprocessors:
            generated_cypher = processor(generated_cypher)
            intermediate_steps.append({type(processor).__name__: generated_cypher})
        self._log_it(_run_manager, generated_cypher)
        return generated_cypher, intermediate_steps

    def _log_it(self, _run_manager: CallbackManagerForChainRun, generated_cypher: str):
        _run_manager.on_text("Preprocessed Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(
            self, inputs: Dict[str, Any], preprocessed_cypher: str, intermediate_steps: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Prepare the chain result.
        """
        chain_result: Dict[str, Any] = {
            self.output_key: preprocessed_cypher
        }

        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key,
                                            []) + intermediate_steps  # inherit intermediate steps from last chain
            chain_result[self.intermediate_steps_key] = intermediate_steps

        return chain_result
