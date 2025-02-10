# !/usr/bin/env python3
# -*- coding:utf-8 -*-


from typing import List, Any, Dict, Tuple, Optional

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks.manager import CallbackManagerForChainRun

from utils.tools import MultihopsExpansion
from utils.utils import graph_result_contains_triple, get_triples_from_graph_result, get_node_name_in_kg


class NodesExpansionChain(Chain):
    """Expand linked entities in knowledge graph

    Args:
        expansioner (MultihopsExpansion): Generate and execute cypher to query multihops from input node
        input_key (str): Input key
        output_key (str): Output key
        cypher_key (str): key of generated cypher query
        limit_record_num (int): Maximum number of response records to reserve
        exclude_types (List[str]): Excluded node types when query
        exclude_props (List[str]): Excluded node properties when query
        return_intermediate_steps (bool): Whether to return intermediate steps
        intermediate_steps_key (str): Key of intermediate steps
    """
    expansioner: MultihopsExpansion
    input_key: str = "linked_entities"
    output_key: str = "node_expand_info"
    cypher_key: str = "expand_cypher_query"
    limited_record_num: int = 50
    exclude_types: List[str] = []
    exclude_props: List[str] = []
    return_intermediate_steps: bool = True
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
            self,
            graph: Neo4jGraph,
            max_hop: int = 2,
            min_hop: int = 1,
            limited_record_num: int = 50,
            exclude_types: List[str] = [],
            exclude_props: List[str] = [],
            return_intermediate_steps: bool = True,
            **kwargs
    ):
        """Init

        Args:
            graph (Neo4jGraph): Graph store
            max_hop (int, optional): Maximal hops of query. Defaults to 1.
            min_hop (int, optional): Minimal hops of query. Defaults to 1.
            limited_record_num(int, optional): limited number of return records from graph. Default to 50
            exclude_types (List[str], optional): Excluded node types when query. Defaults to [].
            exclude_props (List[str], optional): Excluded node properties when query. Defaults to [].
            return_intermediate_steps (bool, optional): Whether to return intermediate steps. Defaults to True.
        """
        expansioner = MultihopsExpansion(graph, max_hop, min_hop)
        super().__init__(
            expansioner=expansioner,
            limited_record_num=limited_record_num,
            exclude_types=exclude_types,
            exclude_props=exclude_props,
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
        return [self.output_key, self.cypher_key]
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        linked_entities: Dict[str, List[Tuple[str, str, str]]] = inputs[self.input_key]
        node_descriptions = []
        cyphers = []
        for entity, linked_nodes in linked_entities.items():
            node = linked_nodes[0]
            e_nodes = self._get_equivalent_node(node)
            expand_infos = []
            for n in [node] + e_nodes:
                expand_info, cypher = self._expand_one_node(n, _run_manager)
                if expand_info:
                    expand_infos.append(expand_info)
                    cyphers.append(cypher)
            
            if expand_infos:
                prefix = f"The entity {entity} mentioned in the question is a {node[0]} type node {node[2]} in the graph. The attribute and relationship information for this node in the graph are as follows:\n"
                node_descriptions.append(prefix + "\n".join(expand_infos))

        return self._prepare_chain_results(inputs, node_descriptions, cyphers)
    
    def _expand_one_node(self, node: Tuple[str, str, str], run_manager: CallbackManagerForChainRun) -> Tuple[str, str]:
        """Expand from one node by query graph and format response"""
        rsp, cypher = self.expansioner.expand(node, return_query=True)
        if len(rsp) == 0:
            return "", cypher

        rsp = rsp[:self.limited_record_num]
        formatted_rsp, err = self._format_graph_response(rsp)
        if err:
            print("Format response cause error {}".format(err))
            run_manager.on_text("Format response cause error:", end='\n', verbose=self.verbose)
            run_manager.on_text(f"{err}", end='\n', color='red', verbose=self.verbose)
        return formatted_rsp, cypher
    
    def _prepare_chain_results(self, inputs: Dict[str, Any], node_descriptions: List[str], cyphers: List[str]):
        """Add output and intermediate record"""
        output: str = "\n".join(node_descriptions)
        chain_result = {self.output_key: output, self.cypher_key: ";\n".join(cyphers)}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result

    def _get_equivalent_node(self, node: Tuple[str, str, str]):
        node_type, node_prop_key, node_prop_val = node
        cypher = f"MATCH (n:{node_type})-[r:semantic]->(e) where n.{node_prop_key} = \"{node_prop_val}\" and r.name=\"equivalent\" return e.name as name"
        e_nodes = self.expansioner._graph.query(cypher)
        return [(node_type, node_prop_key, e["name"]) for e in e_nodes]

    def _format_graph_response(self, rsp: List[Dict[str, Any]]) -> Tuple[str, Optional[Exception]]:
        """format graph response as a text

        Args:
            rsp (List[Dict[str, Any]]): response from query on neo4j graph

        Returns:
            Tuple[str, Optional[Exception]]: formatted text and possible exception
        """
        res_text = ""

        try:
            # extract node info and conver to md table
            node_info = rsp[0]["n"]
            for k in node_info:
                if k in self.exclude_props:
                    node_info.pop(k)
            node_md = "| property | value |\n| --- | --- |\n"
            for k, v in node_info.items():
                node_md += f"| {k} | {v} |\n"
            res_text += node_md
            res_text += "\n"

            # extract triplets and convert to md table
            triplet_set = set()
            for r in rsp:
                if graph_result_contains_triple(r):
                    triplets = get_triples_from_graph_result(r)[0]
                    rel_node_type = r["c_type"]
                    rel_name = r["r_name"]
                    if rel_node_type in self.exclude_types or rel_name == "equivalent":
                        continue
                    head_name = get_node_name_in_kg(triplets[0])
                    tail_name =  get_node_name_in_kg(triplets[2])
                    if head_name and tail_name:
                        triplet_set.add((head_name, rel_name, tail_name))

            triplet_md = "| head node | relation | tail node |\n| --- | --- | --- |\n"
            for h, r, t in triplet_set:
                triplet_md += f"| {h} | {r} | {t} |\n"
            res_text += triplet_md

            return res_text, None

        except Exception as e:
            return res_text, e