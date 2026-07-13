"""
ISCAS-85 Bench file parser.
Parses .bench netlists into a graph representation.
"""

import re
import networkx as nx
from typing import Dict, List, Tuple, Set


GATE_TYPES = {
    'AND': 0, 'NAND': 1, 'OR': 2, 'NOR': 3,
    'XOR': 4, 'XNOR': 5, 'NOT': 6, 'BUFF': 7,
    'BUF': 7, 'INV': 6, 'DFF': 8,
}

GATE_TYPE_REVERSE = {v: k for k, v in GATE_TYPES.items()}


def parse_bench(filepath: str) -> nx.DiGraph:
    """
    Parse a .bench file and return a directed graph.
    Nodes have attributes: type (str), type_id (int), is_input, is_output, is_key.
    Edges go from driver to fanout (data flow direction).
    """
    G = nx.DiGraph()
    inputs: Set[str] = set()
    outputs: Set[str] = set()
    key_inputs: Set[str] = set()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # INPUT declaration
        m = re.match(r'^INPUT\((\w+)\)$', line, re.IGNORECASE)
        if m:
            node = m.group(1)
            inputs.add(node)
            if re.match(r'^key\w*$', node, re.IGNORECASE) or re.match(r'^k\d+$', node, re.IGNORECASE):
                key_inputs.add(node)
            if node not in G:
                G.add_node(node, type='INPUT', type_id=-1,
                           is_input=True, is_output=False, is_key=node in key_inputs)
            continue

        # OUTPUT declaration
        m = re.match(r'^OUTPUT\((\w+)\)$', line, re.IGNORECASE)
        if m:
            node = m.group(1)
            outputs.add(node)
            continue

        # KEY declaration (explicit key-input syntax, e.g. KC2-style locked files)
        m = re.match(r'^KEY\((\w+)\)$', line, re.IGNORECASE)
        if m:
            node = m.group(1)
            inputs.add(node)
            key_inputs.add(node)
            if node not in G:
                G.add_node(node, type='INPUT', type_id=-1,
                           is_input=True, is_output=False, is_key=True)
            else:
                G.nodes[node]['is_key'] = True
                G.nodes[node]['is_input'] = True
            continue

        # Gate assignment: out = GATE(in1, in2, ...)
        m = re.match(r'^(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)$', line, re.IGNORECASE)
        if m:
            out_node = m.group(1)
            gate_type = m.group(2).upper()
            in_nodes = [s.strip() for s in m.group(3).split(',') if s.strip()]
            type_id = GATE_TYPES.get(gate_type, -1)
            G.add_node(out_node, type=gate_type, type_id=type_id,
                       is_input=False, is_output=False, is_key=False)
            for inp in in_nodes:
                if inp not in G:
                    G.add_node(inp, type='WIRE', type_id=-2,
                               is_input=False, is_output=False, is_key=False)
                G.add_edge(inp, out_node)
            continue

    # Mark outputs
    for node in outputs:
        if node in G:
            G.nodes[node]['is_output'] = True

    # Re-mark key inputs (after all nodes are added)
    for node in inputs:
        is_key = (node in key_inputs
                  or node.lower().startswith('key')
                  or re.match(r'^k\d+$', node, re.IGNORECASE))
        if node in G:
            G.nodes[node]['is_key'] = bool(is_key)
            G.nodes[node]['is_input'] = True
            if bool(is_key):
                key_inputs.add(node)

    return G, inputs, outputs, key_inputs


def get_graph_stats(G: nx.DiGraph, key_inputs: Set[str]) -> Dict:
    """Return basic statistics about the parsed netlist."""
    gate_counts = {}
    for node, data in G.nodes(data=True):
        t = data.get('type', 'UNKNOWN')
        gate_counts[t] = gate_counts.get(t, 0) + 1

    return {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'key_inputs': len(key_inputs),
        'gate_counts': gate_counts,
    }