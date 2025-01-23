
import re
import sys
from typing import Tuple, Any, List, Dict, Protocol, runtime_checkable

from functools import partial
from langchain_community.graphs import Neo4jGraph


class MultihopsExpansion:
    """Expand from a single node in mutiple hops"""

    def __init__(self, graph: Neo4jGraph, max_hop: int = 1, min_hop: int = 1):
        self._graph = graph
        query_template = """
        MATCH path = (n: {node_type})-[*{min_hop}..{max_hop}]-(c)
        WHERE n.{node_property}="{node_value}"
        UNWIND relationships(path) as r
        return DISTINCT r, n, c, LABELS(c)[0] as c_type, r.name as r_name
        """
        self._query_gen_func = partial(query_template.format, min_hop=min_hop, max_hop=max_hop)

    def expand(
            self, node: Tuple[str, str, str], return_query=False
    ):
        """Expand from a node in multiple hops"""
        node_type, node_prop_key, node_prop_val = node
        query = self._query_gen_func(
            node_type=node_type,
            node_property=node_prop_key,
            node_value=node_prop_val
        )
        
        results = self._query_graph(query)
     
        if return_query:
            return results, query
        return results

    def _query_graph(self, cypher) -> List[Dict[str, Any]]:
        try:
            return self._graph.query(cypher)
        except Exception as e:
            print(f"Node expansion for {cypher} could not be extracted due to {e}")
        return []
    

@runtime_checkable
class CypherPreprocessor(Protocol):
    def __call__(self, cypher_query: str) -> str:
        ...


class FormatPreprocessor(CypherPreprocessor):
    """A preprocessor applying some regex based formating to cypher queries.
    This allows simpler regex expressions in subsequent preprocessors.
    Partially based on https://github.com/TristanPerry/cypher-query-formatter
    (see MIT License: https://github.com/TristanPerry/cypher-query-formatter/blob/master/LICENSE).
    """

    def __call__(self, cypher_query: str) -> str:
        """Try to format the cypher query.
        If the formatting fails, return the original query.
        """
        try:
            return self._try_formatting(cypher_query)
        except Exception as e:
            print(f"Cypher query formating failed: {e}", file=sys.stderr)
            return cypher_query

    def _try_formatting(self, cypher_query: str) -> str:
        """Formatting pipeline

        Args:
            cypher_query (str): input raw cypher query

        Returns:
            str: a formatted cypher query
        """
        cypher_query = cypher_query.strip()
        cypher_query = self._only_use_double_quotes(cypher_query)
        cypher_query = self._keywords_to_upper_case(cypher_query)
        cypher_query = self._remove_quotes_after_as(cypher_query)
        cypher_query = self._null_and_boolean_literals_to_lower_case(cypher_query)
        cypher_query = self._ensure_main_keywords_on_newline(cypher_query)
        cypher_query = self._unix_style_newlines(cypher_query)
        cypher_query = self._remove_whitespace_from_start_of_lines(cypher_query)
        cypher_query = self._remove_whitespace_from_end_of_lines(cypher_query)
        cypher_query = self._add_spaces_after_comma(cypher_query)
        cypher_query = self._multiple_spaces_to_single_space(cypher_query)
        cypher_query = self._indent_on_create_and_on_match(cypher_query)
        cypher_query = self._remove_multiple_empty_newlines(cypher_query)
        cypher_query = self._remove_unnecessary_spaces(cypher_query)

        return cypher_query.strip()

    def _only_use_double_quotes(self, cypher_query: str) -> str:
        """Escape all single quotes in double quote strings"""
        cypher_query = re.sub(
            r'"([^{}\(\)\[\]=]*?)"', lambda matches: matches.group(0).replace("'", r"\'"), cypher_query
        )
        # Replace all not escaped single quotes.
        cypher_query = re.sub(r"(?<!\\)'", lambda m: m.group(0)[:-1] + '"', cypher_query)
        return cypher_query

    def _keywords_to_upper_case(self, cypher_query: str) -> str:
        """turn all keywords to upper case"""
        return re.sub(
            r"\b(WHEN|CASE|AND|OR|XOR|DISTINCT|AS|IN|STARTS WITH|ENDS WITH|CONTAINS|NOT|SET|ORDER BY)\b",
            _keywords_to_upper_case,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _null_and_boolean_literals_to_lower_case(self, cypher_query: str) -> str:
        """turn all null and boolean literals to lower case"""
        return re.sub(
            r"\b(NULL|TRUE|FALSE)\b",
            _null_and_booleans_to_lower_case,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _ensure_main_keywords_on_newline(self, cypher_query: str) -> str:
        return re.sub(
            r"\b(CASE|DETACH DELETE|DELETE|MATCH|MERGE|LIMIT|OPTIONAL MATCH|RETURN|UNWIND|UNION|WHERE|WITH|GROUP BY)\b",
            _main_keywords_on_newline,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _unix_style_newlines(self, cypher_query: str) -> str:
        return re.sub(r"(\r\n|\r)", "\n", cypher_query)

    def _remove_whitespace_from_start_of_lines(self, cypher_query: str) -> str:
        return re.sub(r"^\s+", "", cypher_query, flags=re.MULTILINE)

    def _remove_whitespace_from_end_of_lines(self, cypher_query: str) -> str:
        return re.sub(r"\s+$", "", cypher_query, flags=re.MULTILINE)

    def _add_spaces_after_comma(self, cypher_query: str) -> str:
        return re.sub(r",([^\s])", lambda matches: ", " + matches.group(1), cypher_query)

    def _multiple_spaces_to_single_space(self, cypher_query: str) -> str:
        return re.sub(r"((?![\n])\s)+", " ", cypher_query)

    def _indent_on_create_and_on_match(self, cypher_query: str) -> str:
        return re.sub(r"\b(ON CREATE|ON MATCH)\b", _indent_on_create_and_on_match, cypher_query, flags=re.IGNORECASE)

    def _remove_multiple_empty_newlines(self, cypher_query: str) -> str:
        return re.sub(r"\n\s*?\n", "\n", cypher_query)

    def _remove_unnecessary_spaces(self, cypher_query: str) -> str:
        cypher_query = re.sub(r"(\(|{|\[])\s+", lambda matches: matches.group(1), cypher_query)
        cypher_query = re.sub(r"\s+(\)|}|\])", lambda matches: matches.group(1), cypher_query)
        cypher_query = re.sub(r"\s*(:|-|>|<)\s*", lambda matches: matches.group(1), cypher_query)
        # Retain spaces before property names
        cypher_query = re.sub(r':\s*"', ': "', cypher_query)
        # Also around equation signs
        cypher_query = re.sub(r"\s+=\s*", " = ", cypher_query)
        cypher_query = re.sub(r"\s*<=\s*", " <= ", cypher_query)
        cypher_query = re.sub(r"\s*>=\s*", " >= ", cypher_query)
        return cypher_query

    def _remove_quotes_after_as(self, cypher_query: str) -> str:
        """remove quotes in the variable names after keyword AS"""
        cypher_query = re.sub(r"(?<=AS)\s*\"(.*?)\"(?=,)", r" \1",
                              cypher_query)  # remove quotes in variable after `AS` except last one
        cypher_query = re.sub(r"(?<=AS)\s*\"(.*?)\"(?=$)", r" \1",
                              cypher_query)  # remove quotes in variable after last `AS`
        return cypher_query


def _keywords_to_upper_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), " " + matches.group(1).upper().strip() + " ")


def _null_and_booleans_to_lower_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), " " + matches.group(1).lower().strip() + " ")


def _main_keywords_on_newline(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), "\n" + matches.group(1).upper().lstrip() + " ")


def _indent_on_create_and_on_match(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), "\n  " + matches.group(1).upper().lstrip() + " ")


class Size2CountPreprocessor(CypherPreprocessor):
    """Replace SIZE keyword with COUNT keyword in Cypher query."""

    def __call__(self, cypher_query: str) -> str:
        return self._replace_size_with_count(cypher_query)

    def _replace_size_with_count(self, cypher_query):
        """replace keyword SIZE with COUNT

        Args:
            cypher_query (_type_): raw cypher query
        """
        cypher_query = re.sub(r"\b(SIZE\b\s*)\(", _size_to_upper_case, cypher_query, flags=re.IGNORECASE)
        search_term = "SIZE("
        len_term = len(search_term)
        while (start := cypher_query.find(search_term)) >= 0:
            end = self._find_closing_bracket(cypher_query, start + len_term)
            cypher_query = (cypher_query[:start] + "COUNT(" + cypher_query[start + len_term: end] + ")" +
                            cypher_query[end + 1:])
        return cypher_query

    def _find_closing_bracket(self, cypher_query: str, search_start: int, bracket_values={"(": 1, ")": -1}) -> int:
        """find the closing bracket of a given cypher query and return the index

        Args:
            cypher_query (str): the given cypher query
            search_start (int): start search position
            bracket_values (dict, optional):  Defaults to {"(": 1, ")": -1}.

        Raises:
            ValueError: cypher grammar error

        Returns:
            int: index of closing braket
        """
        count = 1
        for i, c in enumerate(cypher_query[search_start:]):
            count += bracket_values.get(c, 0)
            if count == 0:
                return i + search_start
        raise ValueError("SIZE keyword in Cypher query without closing bracket.")


def _size_to_upper_case(matches: re.Match[str]) -> str:
    """convert 'size' keyword to upper"""
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), matches.group(1).upper().strip())


class AlwaysDistinctPreprocessor(CypherPreprocessor):
    """Add distinct in the return of the cypher query
    """

    def __call__(self, cypher_query: str) -> str:
        return re.sub(r"RETURN\s+(?!DISTINCT).*", self._replace_match, cypher_query)

    def _replace_match(self, matches: re.Match[str]) -> str:
        return matches.group(0).replace("RETURN", "RETURN DISTINCT")