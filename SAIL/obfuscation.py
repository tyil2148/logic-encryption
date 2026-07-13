"""
Pseudo Self-Referencing training data generation.

Since we don't have access to the original pre-obfuscation netlist,
we treat the obfuscated circuit as a "Pseudo Golden Circuit" and apply
another round of obfuscation to generate [Pre-Synthesis, Post-Synthesis]
locality pairs for training.

Also simulates XOR-based obfuscation and synthesis transformations.
"""

import random
import copy
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from bench_parser import GATE_TYPES


# ─────────────────────────────────────────────
# Synthesis transformation rules (Level-1/2/3)
# ─────────────────────────────────────────────

SYNTHESIS_RULES = {
    # Rule name -> (pre_pattern, post_pattern)
    # These approximate what a synthesis tool does to key-gate localities

    # Level-2: inverter bubble moves from output side to input side of XOR
    'inv_bubble_move': ('XOR+NOT', 'XNOR'),

    # Level-2: XNOR absorbed
    'xnor_absorb': ('XNOR', 'XOR+NOT'),

    # Level-3: XOR absorbed into AND-OR network
    'xor_decompose': ('XOR', 'NAND+NAND+NAND'),

    # Level-2: double inversion cancels
    'double_inv': ('NOT+NOT', 'WIRE'),
}


def simulate_synthesis_transform(gate_sequence: List[str],
                                  rule_prob: float = 0.4) -> Tuple[List[str], int]:
    """
    Simulate what a synthesis tool does to a sequence of gates.
    Returns (transformed_sequence, change_level).

    change_level: 0=no change (Level-1), 1=minor (Level-2), 2=major (Level-3)
    """
    gates = list(gate_sequence)
    changed = False
    change_level = 0

    if random.random() > rule_prob:
        return gates, 0  # Level-1: no change

    # Level-2: inverter moves around XOR gate
    for i in range(len(gates) - 1):
        if gates[i] == 'XOR' and gates[i + 1] == 'NOT':
            gates[i] = 'XNOR'
            gates.pop(i + 1)
            change_level = 1
            changed = True
            break
        elif gates[i] == 'NOT' and gates[i + 1] == 'XOR':
            gates[i] = 'XOR'
            gates[i + 1] = 'NOT'
            change_level = 1
            changed = True
            break

    # Level-3: XOR decomposed into NAND tree
    if not changed and random.random() < 0.2:
        for i, g in enumerate(gates):
            if g == 'XOR':
                gates[i:i + 1] = ['NAND', 'NAND', 'NAND']
                change_level = 2
                break

    return gates, change_level


def insert_xor_obfuscation(G: nx.DiGraph, target_node: str,
                            key_bit: int = 0) -> Tuple[nx.DiGraph, str, str]:
    """
    Insert XOR-based obfuscation at target_node.
    - keybit=0: insert XOR gate
    - keybit=1: insert a raw XNOR gate (the common convention used by real
      locking tools -- EPIC, SARLock, and random XOR/XNOR locking all wire a
      single XNOR cell rather than a separate XOR+inverter pair), or,
      occasionally, XOR followed by a separate inverter (the paper's Fig. 3b
      pre-synthesis form) so the reconstruction model sees both variants.

    Returns (new_graph, key_node_name, center_node_name)
    """
    G_new = copy.deepcopy(G)

    key_node = f"key_{target_node}_{random.randint(1000, 9999)}"
    xor_node = f"xor_obf_{target_node}"
    inv_node = f"inv_obf_{target_node}"

    G_new.add_node(key_node, type='INPUT', type_id=-1,
                   is_input=True, is_output=False, is_key=True)

    # Get predecessors of target
    preds = list(G_new.predecessors(target_node))

    if key_bit == 1 and random.random() < 0.5:
        # Raw XNOR gate directly on the key wire (matches real lockers)
        G_new.add_node(xor_node, type='XNOR', type_id=GATE_TYPES['XNOR'],
                       is_input=False, is_output=False, is_key=False)
        if preds:
            first_pred = preds[0]
            G_new.remove_edge(first_pred, target_node)
            G_new.add_edge(first_pred, xor_node)
        G_new.add_edge(key_node, xor_node)
        G_new.add_edge(xor_node, target_node)
        return G_new, key_node, xor_node

    G_new.add_node(xor_node, type='XOR', type_id=GATE_TYPES['XOR'],
                   is_input=False, is_output=False, is_key=False)

    # Wire: first predecessor -> xor_node, key -> xor_node
    if preds:
        first_pred = preds[0]
        G_new.remove_edge(first_pred, target_node)
        G_new.add_edge(first_pred, xor_node)
    G_new.add_edge(key_node, xor_node)

    if key_bit == 1:
        # Add inverter after XOR (unabsorbed pre-synthesis form)
        G_new.add_node(inv_node, type='NOT', type_id=GATE_TYPES['NOT'],
                       is_input=False, is_output=False, is_key=False)
        G_new.add_edge(xor_node, inv_node)
        G_new.add_edge(inv_node, target_node)
        return G_new, key_node, inv_node
    else:
        G_new.add_edge(xor_node, target_node)
        return G_new, key_node, xor_node


def apply_synthesis_to_graph(G: nx.DiGraph, key_nodes: Set[str],
                              change_prob: float = 0.6) -> Tuple[nx.DiGraph, Dict]:
    """
    Apply synthesis-like transformations to the graph around key nodes.
    Returns (transformed_graph, change_log).
    """
    G_synth = copy.deepcopy(G)
    change_log = {}

    for key_node in key_nodes:
        if key_node not in G_synth:
            continue
        succs = list(G_synth.successors(key_node))
        for xor_gate in succs:
            if G_synth.nodes[xor_gate].get('type', '') not in ('XOR', 'XNOR'):
                continue
            r = random.random()
            if r > change_prob:
                change_log[xor_gate] = ('level1', 0)
                continue

            # Level-2: inverter bubble move
            xor_succs = list(G_synth.successors(xor_gate))
            inv_succs = [s for s in xor_succs
                         if G_synth.nodes[s].get('type', '') in ('NOT', 'INV')]
            if inv_succs and random.random() < 0.6:
                # Transform XOR+NOT -> XNOR (inverter absorbed)
                inv_node = inv_succs[0]
                inv_out_nodes = list(G_synth.successors(inv_node))
                G_synth.nodes[xor_gate]['type'] = 'XNOR'
                G_synth.nodes[xor_gate]['type_id'] = GATE_TYPES['XNOR']
                # Re-wire: xor_gate -> downstream of inv_node
                G_synth.remove_node(inv_node)
                for out_n in inv_out_nodes:
                    if out_n in G_synth:
                        G_synth.add_edge(xor_gate, out_n)
                change_log[xor_gate] = ('level2', 1)

            elif random.random() < 0.15:
                # Level-3: decompose XOR into 3-NAND structure
                preds = list(G_synth.predecessors(xor_gate))
                out_nodes = list(G_synth.successors(xor_gate))
                if len(preds) >= 2:
                    n1 = f"nand1_{xor_gate}"
                    n2 = f"nand2_{xor_gate}"
                    n3 = f"nand3_{xor_gate}"
                    for n, t in [(n1, 'NAND'), (n2, 'NAND'), (n3, 'NAND')]:
                        G_synth.add_node(n, type=t, type_id=GATE_TYPES['NAND'],
                                         is_input=False, is_output=False, is_key=False)
                    G_synth.add_edge(preds[0], n1)
                    G_synth.add_edge(preds[1], n1)
                    G_synth.add_edge(preds[0], n2)
                    G_synth.add_edge(n1, n2)
                    G_synth.add_edge(preds[1], n3)  # simplified
                    G_synth.add_edge(n1, n3)
                    G_synth.remove_node(xor_gate)
                    for out_n in out_nodes:
                        if out_n in G_synth:
                            G_synth.add_edge(n3, out_n)
                    change_log[n3] = ('level3', 2)
            else:
                change_log[xor_gate] = ('level1', 0)

    return G_synth, change_log


def generate_pseudo_self_reference_pairs(
        G_obfuscated: nx.DiGraph,
        key_inputs: Set[str],
        n_iterations: int = 5,
        keys_per_iter: int = 8
) -> List[Tuple[nx.DiGraph, nx.DiGraph, str, int]]:
    """
    Pseudo Self-Referencing: treat G_obfuscated as golden, apply another
    round of obfuscation + synthesis to generate training pairs.

    Returns list of (pre_synthesis_locality, post_synthesis_locality,
                     center_node, change_level)
    """
    pairs = []
    candidate_nodes = [
        n for n, d in G_obfuscated.nodes(data=True)
        if d.get('type', '') not in ('INPUT', 'WIRE', 'OUTPUT', '')
        and not d.get('is_key', False)
    ]

    if not candidate_nodes:
        return pairs

    for _ in range(n_iterations):
        chosen = random.sample(candidate_nodes,
                               min(keys_per_iter, len(candidate_nodes)))
        G_pre = copy.deepcopy(G_obfuscated)
        new_key_nodes = set()

        for node in chosen:
            if node not in G_pre:
                continue
            kb = random.randint(0, 1)
            try:
                G_pre, kn, _ = insert_xor_obfuscation(G_pre, node, key_bit=kb)
                new_key_nodes.add(kn)
            except Exception:
                continue

        G_post, change_log = apply_synthesis_to_graph(G_pre, new_key_nodes)

        # Extract localities around new key-gate nodes
        for key_n in new_key_nodes:
            if key_n not in G_pre or key_n not in G_post:
                continue
            succs_pre = list(G_pre.successors(key_n))
            succs_post = list(G_post.successors(key_n))
            center_candidates = succs_pre + succs_post
            if not center_candidates:
                continue
            center = center_candidates[0]

            # Extract 5-hop locality
            def get_locality(G, c, radius=5):
                try:
                    G_u = G.to_undirected()
                    if c not in G_u:
                        return G.subgraph([]).copy()
                    nodes = nx.ego_graph(G_u, c, radius=radius).nodes()
                    return G.subgraph(nodes).copy()
                except Exception:
                    return G.subgraph([]).copy()

            loc_pre = get_locality(G_pre, center)
            loc_post = get_locality(G_post, center)

            cl_info = change_log.get(center, change_log.get(
                next(iter(change_log), None), ('level1', 0)))
            cl = {'level1': 0, 'level2': 1, 'level3': 2}.get(cl_info[0], 0)

            pairs.append((loc_pre, loc_post, center, cl))

    return pairs