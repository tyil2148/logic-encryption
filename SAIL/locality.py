"""
Locality extraction and feature engineering for SAIL.

A "locality" is the subgraph of up to k gates surrounding a key-gate.
Features are extracted from these subgraphs for the ML models.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from bench_parser import GATE_TYPES


def extract_locality(G: nx.DiGraph, center_node: str, k: int = 3) -> nx.DiGraph:
    """
    Extract a k-hop ego subgraph centered on center_node.
    k is the radius in terms of graph hops (undirected).
    """
    # Use undirected version for neighborhood extraction
    G_undirected = G.to_undirected()
    nodes_in_locality = set(nx.ego_graph(G_undirected, center_node, radius=k).nodes())
    subgraph = G.subgraph(nodes_in_locality).copy()
    return subgraph


def locality_to_feature_vector(subgraph: nx.DiGraph, center_node: str,
                                 max_nodes: int = 20) -> np.ndarray:
    """
    Convert a locality subgraph into a fixed-length feature vector.

    Features include:
    - Gate type histogram (normalized)
    - Degree statistics of the center node and neighbors
    - Structural metrics (density, clustering, etc.)
    - Key connectivity features
    """
    num_gate_types = len(GATE_TYPES) + 3  # +3 for INPUT, WIRE, UNKNOWN

    # Gate type histogram
    type_hist = np.zeros(num_gate_types)
    for node, data in subgraph.nodes(data=True):
        tid = data.get('type_id', -2)
        if tid >= 0:
            type_hist[tid] += 1
        elif data.get('is_input', False):
            type_hist[num_gate_types - 2] += 1
        else:
            type_hist[num_gate_types - 1] += 1

    total = type_hist.sum()
    if total > 0:
        type_hist_norm = type_hist / total
    else:
        type_hist_norm = type_hist

    # Center node features
    center_in_deg = subgraph.in_degree(center_node) if center_node in subgraph else 0
    center_out_deg = subgraph.out_degree(center_node) if center_node in subgraph else 0
    center_type_id = subgraph.nodes[center_node].get('type_id', -1) if center_node in subgraph else -1
    center_is_key = float(subgraph.nodes[center_node].get('is_key', False)) if center_node in subgraph else 0.0

    # Normalize center type
    center_type_onehot = np.zeros(num_gate_types)
    if 0 <= center_type_id < num_gate_types:
        center_type_onehot[center_type_id] = 1.0

    # Subgraph statistics
    n_nodes = subgraph.number_of_nodes()
    n_edges = subgraph.number_of_edges()
    density = nx.density(subgraph) if n_nodes > 1 else 0.0

    # Degree statistics across all nodes
    in_degrees = [subgraph.in_degree(n) for n in subgraph.nodes()]
    out_degrees = [subgraph.out_degree(n) for n in subgraph.nodes()]

    in_deg_mean = np.mean(in_degrees) if in_degrees else 0
    in_deg_std = np.std(in_degrees) if in_degrees else 0
    out_deg_mean = np.mean(out_degrees) if out_degrees else 0
    out_deg_std = np.std(out_degrees) if out_degrees else 0

    # Key nodes in locality
    key_count = sum(1 for _, d in subgraph.nodes(data=True) if d.get('is_key', False))
    xor_count = sum(1 for _, d in subgraph.nodes(data=True) if d.get('type', '') in ('XOR', 'XNOR'))
    inv_count = sum(1 for _, d in subgraph.nodes(data=True) if d.get('type', '') in ('NOT', 'INV'))

    # Assemble feature vector
    scalar_features = np.array([
        center_in_deg, center_out_deg, center_is_key,
        n_nodes / max_nodes,  # normalized
        n_edges / (max_nodes * 2),
        density,
        in_deg_mean, in_deg_std,
        out_deg_mean, out_deg_std,
        key_count / max(n_nodes, 1),
        xor_count / max(n_nodes, 1),
        inv_count / max(n_nodes, 1),
    ], dtype=np.float32)

    feature_vector = np.concatenate([
        type_hist_norm.astype(np.float32),
        center_type_onehot.astype(np.float32),
        scalar_features
    ])

    return feature_vector


def locality_to_label(subgraph_pre: nx.DiGraph, subgraph_post: nx.DiGraph,
                       center_node: str) -> Tuple[int, int, int]:
    """
    Compute change label comparing pre- and post-synthesis localities.

    Returns:
        change_level: 0 = no change (Level-1), 1 = partial (Level-2), 2 = full transform (Level-3)
        gate_error: number of gate type mismatches in the 3-gate snapshot
        link_error: number of edge mismatches
    """
    # Gate type comparison
    pre_types = {n: d.get('type', '') for n, d in subgraph_pre.nodes(data=True)}
    post_types = {n: d.get('type', '') for n, d in subgraph_post.nodes(data=True)}

    common_nodes = set(pre_types.keys()) & set(post_types.keys())
    gate_errors = sum(1 for n in common_nodes if pre_types[n] != post_types[n])
    added_nodes = len(set(post_types.keys()) - set(pre_types.keys()))
    removed_nodes = len(set(pre_types.keys()) - set(post_types.keys()))
    gate_error = gate_errors + added_nodes + removed_nodes

    # Edge comparison
    pre_edges = set(subgraph_pre.edges())
    post_edges = set(subgraph_post.edges())
    link_error = len(pre_edges.symmetric_difference(post_edges))

    # Determine change level
    if gate_error == 0 and link_error == 0:
        change_level = 0  # Level-1: no change
    elif gate_error <= 2 and link_error <= 2:
        change_level = 1  # Level-2: minor change
    else:
        change_level = 2  # Level-3: major transform

    return change_level, gate_error, link_error


def get_key_gate_neighbors(G: nx.DiGraph, key_inputs: Set[str]) -> List[str]:
    """
    Find all gate nodes that are directly driven by a key input wire.
    These are the 'key-gate localities' to analyze.
    """
    key_gate_centers = []
    for key_node in key_inputs:
        if key_node in G:
            for successor in G.successors(key_node):
                data = G.nodes[successor]
                if data.get('type', '') not in ('INPUT', 'WIRE', ''):
                    key_gate_centers.append(successor)
    return list(set(key_gate_centers))


def extract_all_localities(G: nx.DiGraph, centers: List[str],
                            locality_size: int = 5) -> List[Tuple[str, nx.DiGraph]]:
    """Extract localities for all center nodes."""
    localities = []
    for center in centers:
        if center in G:
            subgraph = extract_locality(G, center, k=locality_size)
            localities.append((center, subgraph))
    return localities


def infer_key_bit(G: nx.DiGraph, key_node: str,
                   reconstructed_type: Optional[str] = None) -> Optional[int]:
    """
    Infer the XOR-locking key bit for a key input (paper Sec. VI-B.1): a bare
    XOR gate on the key wire encodes key-bit 0, a bare XNOR gate encodes key-bit 1.

    Real locking tools (KC2/random XOR-XNOR, EPIC, SARLock) all insert the key
    gate this way -- gate type alone carries the bit, with no separate trailing
    inverter. An earlier version of this function also treated a dedicated
    inverter immediately downstream of an XOR as key-bit 1 (matching the
    "XOR+NOT" idiom described in the paper), but that pattern doesn't appear in
    any of these real tools and it produced false positives whenever the
    circuit's own next gate happened to be a NOT unrelated to the key -- so it
    was dropped.

    `reconstructed_type` lets the caller pass SAIL's recovered pre-synthesis gate
    type for the center node (e.g. after a Level-3 decomposition obscured it),
    instead of relying on the raw post-synthesis type.
    """
    if key_node not in G:
        return None
    succs = [s for s in G.successors(key_node)
             if G.nodes[s].get('type', '') not in ('INPUT', 'WIRE', '')]
    if not succs:
        return None
    center = succs[0]
    gate_type = (reconstructed_type or G.nodes[center].get('type', '')).upper()

    if gate_type == 'XNOR':
        return 1
    if gate_type == 'XOR':
        return 0
    return None  # inconclusive: not an XOR/XNOR-family key gate


def subgraph_snapshot(subgraph: nx.DiGraph, center_node: str,
                       output_locality_size: int = 3) -> nx.DiGraph:
    """
    Extract a smaller 'snapshot' of size output_locality_size from the locality.
    This is the output representation used for recovery evaluation.
    """
    G_undirected = subgraph.to_undirected()
    if center_node not in G_undirected:
        return subgraph
    snap_nodes = set(nx.ego_graph(G_undirected, center_node,
                                   radius=output_locality_size).nodes())
    return subgraph.subgraph(snap_nodes).copy()