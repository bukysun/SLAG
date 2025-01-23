# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from typing import List, Dict, Optional, Any

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun


class CypherQueryChain(Chain):
    """Excute cypher query on neo4j graph

    Args:
        graph: Neo4j graph store
        topk: results
        return_intermediate_steps: whether to return intermediate steps
        intermediate_steps_key: key for intermediate steps
        input_key: key for input
        output_key: key for output
    """
    graph: Neo4jGraph
    top_k: int = 50
    return_intermediate_steps: bool = True
    intermediate_steps_key: str = "intermediate_steps"
    input_key: str = "preprocessed_cypher_query"
    output_key: str = "cypher_query_result"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys"""
        return [self.input_key]
       
    @property
    def output_keys(self) -> List[str]:
        """Return the output keys"""
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        execute_cypher = inputs[self.input_key]
        rsp = self._query_graph(execute_cypher, _run_manager)
        self._log_it(_run_manager, rsp)
        return self._prepare_chain_result(inputs, rsp)

    def _query_graph(self, cypher: str, run_manager: CallbackManagerForChainRun) -> List[Dict[str, Any]]:
        """Execute cypher on graph"""
        if cypher and 'SCHEMA_ERROR' not in cypher:
            try:
                return self.graph.query(cypher)[: self.top_k]
            except Exception as e:
                run_manager.on_text(f"execute cypher {cypher} casue error:", end='\n', verbose=self.verbose)
                run_manager.on_text(f"{e}", end='\n', color='red', verbose=self.verbose)
        return []

    def _log_it(self, run_manager, query_result):
        run_manager.on_text("Graph Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(query_result), color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(self, inputs: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add output and intermediate record"""
        chain_result = {
            self.output_key: results
        }
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, []) + [{self.output_key: results}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
