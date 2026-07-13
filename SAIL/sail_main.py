"""
SAIL: Structural Analysis using Machine Learning — Main Runner

Usage:
    python sail_main.py --bench path/to/circuit.bench [options]
    python sail_main.py --bench_dir path/to/iscas/ [options]

Options:
    --bench         Path to a single .bench file
    --bench_dir     Directory containing multiple .bench files
    --key_size      Number of key bits to insert (default: 32)
    --locality_size Input locality size for feature extraction (default: 6)
    --iterations    Number of pseudo-self-reference iterations (default: 10)
    --output        Output report file (default: sail_results.txt)
    --verbose       Print detailed per-benchmark results
"""

import os
import re
import sys
import time
import random
import argparse
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── SAIL modules ──────────────────────────────────────────────────────────────
from bench_parser import parse_bench, get_graph_stats, GATE_TYPE_REVERSE
from locality import (extract_locality, get_key_gate_neighbors,
                      extract_all_localities, infer_key_bit)
from obfuscation import (insert_xor_obfuscation, apply_synthesis_to_graph,
                         generate_pseudo_self_reference_pairs)
from models import ChangePredictionModel, ReconstructionEnsemble, SAILModel


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    banner = """
SAIL
"""
    print(banner)


def format_table(headers: List[str], rows: List[List], col_widths: List[int] = None) -> str:
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    sep = '+' + '+'.join('-' * w for w in col_widths) + '+'
    fmt = '|' + '|'.join(f'{{:^{w}}}' for w in col_widths) + '|'
    lines = [sep, fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(x) for x in row]))
    lines.append(sep)
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-benchmark analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_benchmark(bench_file: str,
                       key_size: int = 32,
                       locality_size: int = 6,
                       n_iterations: int = 10,
                       verbose: bool = False) -> Dict:
    """
    Run the full SAIL attack pipeline on one benchmark.

    Steps:
    1. Parse the .bench netlist
    2. Simulate XOR obfuscation to create "obfuscated" design
    3. Generate Pseudo-Self-Reference training pairs
    4. Train Change Prediction + Reconstruction models
    5. Run attack on test localities; compute metrics
    """
    name = os.path.splitext(os.path.basename(bench_file))[0]
    if verbose:
        print(f"\n{'-'*60}")
        print(f"Benchmark: {name}")
        print(f"{'-'*60}")

    t_start = time.time()

    # ── Step 1: Parse ──────────────────────────────────────────────────────
    try:
        G, inputs, outputs, key_inputs_orig = parse_bench(bench_file)
    except Exception as e:
        return {'benchmark': name, 'error': str(e)}

    stats = get_graph_stats(G, key_inputs_orig)
    if verbose:
        print(f"Parsed: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        print(f"Existing key inputs found: {stats['key_inputs']}")
        gc = {k: v for k, v in stats['gate_counts'].items() if v > 0}
        print(f"Gate counts: {gc}")

    # ── Step 2: Simulate obfuscation ────────────────────────────────────────
    candidate_gates = [
        n for n, d in G.nodes(data=True)
        if d.get('type', '') not in ('INPUT', 'WIRE', 'OUTPUT', '')
        and not d.get('is_key', False)
    ]
    if not candidate_gates:
        return {'benchmark': name, 'error': 'No candidate gates found for obfuscation'}

    actual_key_size = min(key_size, len(candidate_gates))
    chosen_gates = random.sample(candidate_gates, actual_key_size)

    if verbose:
        print(f"Inserting {actual_key_size}-bit XOR obfuscation...")

    G_obf = G.copy()
    inserted_key_nodes = set()
    for gate in chosen_gates:
        if gate not in G_obf:
            continue
        kb = random.randint(0, 1)
        try:
            G_obf, kn, _ = insert_xor_obfuscation(G_obf, gate, key_bit=kb)
            inserted_key_nodes.add(kn)
        except Exception:
            continue

    # Apply synthesis transformations
    G_synth, change_log = apply_synthesis_to_graph(G_obf, inserted_key_nodes, change_prob=0.6)

    # Characterize changes
    level_counts = {0: 0, 1: 0, 2: 0}
    for _, (lvl_str, _) in change_log.items():
        lv = {'level1': 0, 'level2': 1, 'level3': 2}.get(lvl_str, 0)
        level_counts[lv] += 1
    total_logged = sum(level_counts.values())
    if total_logged == 0:
        total_logged = 1  # avoid div/0

    if verbose:
        print(f"Change analysis: Level-1={level_counts[0]}({level_counts[0]/total_logged*100:.1f}%), "
              f"Level-2={level_counts[1]}({level_counts[1]/total_logged*100:.1f}%), "
              f"Level-3={level_counts[2]}({level_counts[2]/total_logged*100:.1f}%)")

    # ── Step 3: Generate training data (Pseudo Self-Reference) ──────────────
    if verbose:
        print(f"Generating pseudo-self-reference training pairs "
              f"({n_iterations} iterations)...")

    all_key_inputs = inserted_key_nodes | key_inputs_orig
    training_pairs = generate_pseudo_self_reference_pairs(
        G_obf, all_key_inputs,
        n_iterations=n_iterations,
        keys_per_iter=min(actual_key_size, 16)
    )

    if verbose:
        print(f"Generated {len(training_pairs)} training pairs")

    if len(training_pairs) < 4:
        return {
            'benchmark': name,
            'error': f'Insufficient training data ({len(training_pairs)} pairs). '
                     'Try a larger benchmark or more iterations.'
        }

    # Split train / test
    random.shuffle(training_pairs)
    split = max(1, int(0.75 * len(training_pairs)))
    train_pairs = training_pairs[:split]
    test_pairs = training_pairs[split:]

    if not test_pairs:
        test_pairs = training_pairs[-max(1, len(training_pairs)//4):]

    # ── Step 4: Train SAIL model ────────────────────────────────────────────
    if verbose:
        print(f"Training SAIL model ({len(train_pairs)} train / "
              f"{len(test_pairs)} test samples)...")

    sail = SAILModel()
    sail.fit(train_pairs)

    # ── Step 5: Attack & evaluate ────────────────────────────────────────────
    post_test = [(post, center) for pre, post, center, cl in test_pairs]
    pre_test  = [(pre,  center) for pre, post, center, cl in test_pairs]

    results_dict = sail.attack(post_test, pre_localities=pre_test)
    metrics = results_dict.get('metrics', {})

    elapsed = time.time() - t_start

    result = {
        'benchmark': name,
        'nodes': stats['total_nodes'],
        'edges': stats['total_edges'],
        'key_size': actual_key_size,
        'train_samples': len(train_pairs),
        'test_samples': len(test_pairs),
        'level1_pct': level_counts[0] / total_logged * 100,
        'level2_pct': level_counts[1] / total_logged * 100,
        'level3_pct': level_counts[2] / total_logged * 100,
        'complete_recovery_pct': metrics.get('complete_recovery_pct', 0.0),
        'r_metric': metrics.get('r_metric', 0.0),
        'gate_error_0': metrics.get('gate_error_0', 0),
        'gate_error_1': metrics.get('gate_error_1', 0),
        'gate_error_2': metrics.get('gate_error_2', 0),
        'n_predicted_changed': results_dict.get('n_predicted_changed', 0),
        'n_predicted_unchanged': results_dict.get('n_predicted_unchanged', 0),
        'elapsed_sec': elapsed,
    }

    if verbose:
        m = result
        print(f"\nResults:")
        print(f"  Complete Recovery (G=0,L=0): {m['complete_recovery_pct']:.2f}%")
        print(f"  R-Metric:                    {m['r_metric']:.2f}%")
        print(f"  Gate Error 0/1/2:            {m['gate_error_0']} / "
              f"{m['gate_error_1']} / {m['gate_error_2']}")
        print(f"  Predicted changed:           {m['n_predicted_changed']} / "
              f"{m['test_samples']}")
        print(f"  Time: {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Locked-netlist analysis (attack a real, already-obfuscated .bench file)
# ─────────────────────────────────────────────────────────────────────────────

def load_key_file(key_file: str) -> Dict[str, int]:
    """
    Parse a ground-truth key file (`name = 0/1` per line, `#` comments allowed).
    Used only to score the attack's key guesses -- SAIL itself never reads this.
    """
    gt = {}
    with open(key_file, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            m = re.match(r'^(\w+)\s*=\s*([01])$', line)
            if m:
                gt[m.group(1)] = int(m.group(2))
    return gt


def attack_locked_benchmark(bench_file: str,
                             key_file: Optional[str] = None,
                             extra_key_size: int = 16,
                             locality_size: int = 6,
                             n_iterations: int = 10,
                             verbose: bool = False) -> Dict:
    """
    Run SAIL against an already-locked netlist (real key gates, unknown
    pre-obfuscation structure) -- no golden/unlocked reference used.

    Steps:
    1. Parse the locked .bench file as-is; its key inputs are the real attack targets.
    2. Pseudo Self-Reference (paper Sec. IV-A): treat the locked netlist as a
       pseudo-golden circuit, insert one more synthetic round of XOR/XNOR key-gates
       elsewhere, and run synthesis simulation to build [pre,post] training pairs.
    3. Train the Change Prediction + Reconstruction Ensemble on those pairs, and
       report R-Metric/complete-recovery on a held-out synthetic split (there is
       no ground truth for the real target, same as in the paper's threat model).
    4. Run the trained SAIL model on the localities around the REAL key gates to
       recover their pre-synthesis structure, and infer each key bit from the
       recovered XOR/XNOR pattern.
    5. If a ground-truth key file is supplied, score the inferred bits against it
       (purely for validating this tool -- not part of the attack itself).
    """
    name = os.path.splitext(os.path.basename(bench_file))[0]
    if verbose:
        print(f"\n{'-'*60}")
        print(f"Locked benchmark: {name}")
        print(f"{'-'*60}")

    t_start = time.time()

    try:
        G, inputs, outputs, key_inputs = parse_bench(bench_file)
    except Exception as e:
        return {'benchmark': name, 'error': str(e)}

    if not key_inputs:
        return {'benchmark': name, 'error': 'No key inputs detected in this netlist '
                                             '(expected INPUT(key*)/INPUT(keyinputN) or KEY(...) declarations)'}

    stats = get_graph_stats(G, key_inputs)
    if verbose:
        print(f"Parsed: {stats['total_nodes']} nodes, {stats['total_edges']} edges, "
              f"{len(key_inputs)} real key inputs")

    # ── Step 2: Pseudo Self-Reference training data ─────────────────────────
    candidate_gates = [
        n for n, d in G.nodes(data=True)
        if d.get('type', '') not in ('INPUT', 'WIRE', 'OUTPUT', '')
        and not d.get('is_key', False)
    ]
    if not candidate_gates:
        return {'benchmark': name, 'error': 'No non-key gates available to build the '
                                             'pseudo-self-reference training set'}

    actual_extra_key_size = min(extra_key_size, len(candidate_gates))
    if verbose:
        print(f"Generating pseudo-self-reference training pairs "
              f"({n_iterations} iterations, {actual_extra_key_size} synthetic keys/iter)...")

    training_pairs = generate_pseudo_self_reference_pairs(
        G, key_inputs, n_iterations=n_iterations,
        keys_per_iter=min(actual_extra_key_size, 16)
    )

    if len(training_pairs) < 4:
        return {'benchmark': name,
                'error': f'Insufficient training data ({len(training_pairs)} pairs) '
                         'from pseudo-self-reference. Try more iterations.'}

    random.shuffle(training_pairs)
    split = max(1, int(0.75 * len(training_pairs)))
    train_pairs = training_pairs[:split]
    test_pairs = training_pairs[split:] or training_pairs[-max(1, len(training_pairs)//4):]

    if verbose:
        print(f"Training SAIL model ({len(train_pairs)} train / {len(test_pairs)} test samples)...")

    sail = SAILModel()
    sail.fit(train_pairs)

    post_test = [(post, center) for pre, post, center, cl in test_pairs]
    pre_test = [(pre, center) for pre, post, center, cl in test_pairs]
    val_results = sail.attack(post_test, pre_localities=pre_test)
    val_metrics = val_results.get('metrics', {})

    # ── Step 3: Attack the REAL key-gate localities ─────────────────────────
    key_to_center: Dict[str, str] = {}
    for key_node in key_inputs:
        if key_node not in G:
            continue
        succs = [s for s in G.successors(key_node)
                 if G.nodes[s].get('type', '') not in ('INPUT', 'WIRE', '')]
        if succs:
            key_to_center[key_node] = succs[0]

    real_centers = sorted(set(key_to_center.values()))
    real_localities = extract_all_localities(G, real_centers, locality_size=locality_size)
    center_to_idx = {c: i for i, (c, _) in enumerate(real_localities)}
    post_localities_real = [(subgraph, center) for center, subgraph in real_localities]

    real_results = sail.attack(post_localities_real)
    reconstructed_types = real_results.get('reconstructed_gate_types', {})

    ground_truth = load_key_file(key_file) if key_file else None

    rows = []
    for key_node, center in sorted(key_to_center.items()):
        idx = center_to_idx.get(center)
        raw_type = G.nodes[center].get('type', '?')
        recon_type_id = reconstructed_types.get(idx, -1) if idx is not None else -1
        recon_type = GATE_TYPE_REVERSE.get(recon_type_id, raw_type)
        # Trust the reconstruction only when it proposes a plausible XOR-locking
        # key-gate shape; a reconstruction that collapses to a non-XOR/XNOR gate
        # (e.g. AND/NAND) is more likely a reconstruction miss than a genuine
        # recovered structure, so fall back to the as-observed type -- which is
        # already correct whenever this locality was never resynthesized.
        effective_type = recon_type if recon_type in ('XOR', 'XNOR') else raw_type
        inferred_bit = infer_key_bit(G, key_node, reconstructed_type=effective_type)
        actual_bit = ground_truth.get(key_node) if ground_truth else None
        rows.append({
            'key': key_node, 'center': center,
            'raw_type': raw_type, 'recovered_type': effective_type,
            'inferred_key': inferred_bit, 'actual_key': actual_bit,
        })

    resolvable = [r for r in rows if r['actual_key'] is not None and r['inferred_key'] is not None]
    matches = [r for r in resolvable if r['inferred_key'] == r['actual_key']]
    key_accuracy = (len(matches) / len(resolvable) * 100) if resolvable else None

    elapsed = time.time() - t_start

    result = {
        'benchmark': name,
        'nodes': stats['total_nodes'],
        'edges': stats['total_edges'],
        'n_real_keys': len(key_inputs),
        'n_resolved_gates': len(real_centers),
        'train_samples': len(train_pairs),
        'test_samples': len(test_pairs),
        'complete_recovery_pct': val_metrics.get('complete_recovery_pct', 0.0),
        'r_metric': val_metrics.get('r_metric', 0.0),
        'key_rows': rows,
        'key_accuracy_pct': key_accuracy,
        'n_key_scored': len(resolvable),
        'elapsed_sec': elapsed,
    }

    if verbose:
        print(f"\nValidation (held-out synthetic pairs): "
              f"Recovery={result['complete_recovery_pct']:.1f}%  R-Metric={result['r_metric']:.1f}%")
        print(f"\nReal key-gate recovery ({len(rows)} key inputs):")
        headers = ['Key Input', 'Target Gate', 'Raw Type', 'Recovered Type', 'Inferred Key', 'Actual Key']
        table_rows = [[r['key'], r['center'], r['raw_type'], r['recovered_type'],
                       '?' if r['inferred_key'] is None else r['inferred_key'],
                       '-' if r['actual_key'] is None else r['actual_key']] for r in rows]
        print(format_table(headers, table_rows))
        if key_accuracy is not None:
            print(f"\nKey-recovery accuracy vs. ground truth: {key_accuracy:.1f}% "
                  f"({len(matches)}/{len(resolvable)} scorable bits)")
        print(f"Time: {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-benchmark driver
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(bench_files: List[str],
                   key_size: int = 32,
                   locality_size: int = 6,
                   n_iterations: int = 10,
                   output_file: str = 'sail_results.txt',
                   verbose: bool = False) -> List[Dict]:

    all_results = []
    for bf in bench_files:
        print(f"\n[*] Processing: {os.path.basename(bf)}")
        result = analyze_benchmark(bf, key_size=key_size,
                                    locality_size=locality_size,
                                    n_iterations=n_iterations,
                                    verbose=verbose)
        all_results.append(result)
        if 'error' in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    Recovery: {result['complete_recovery_pct']:.1f}% | "
                  f"R-Metric: {result['r_metric']:.1f}% | "
                  f"Time: {result['elapsed_sec']:.1f}s")

    # Summary report
    good = [r for r in all_results if 'error' not in r]
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SAIL Attack Results Summary")
    report_lines.append("=" * 70)

    if good:
        headers = ['Benchmark', 'Nodes', 'KeySz', 'L1%', 'L2%', 'L3%',
                   'Recov%', 'R-Metric', 'Time(s)']
        rows = []
        for r in good:
            rows.append([
                r['benchmark'], r['nodes'], r['key_size'],
                f"{r['level1_pct']:.0f}", f"{r['level2_pct']:.0f}",
                f"{r['level3_pct']:.0f}",
                f"{r['complete_recovery_pct']:.1f}",
                f"{r['r_metric']:.1f}",
                f"{r['elapsed_sec']:.1f}"
            ])

        # Averages
        avg_rec = np.mean([r['complete_recovery_pct'] for r in good])
        avg_r   = np.mean([r['r_metric'] for r in good])
        avg_t   = np.mean([r['elapsed_sec'] for r in good])
        rows.append(['AVERAGE', '-', '-', '-', '-', '-',
                     f"{avg_rec:.1f}", f"{avg_r:.1f}", f"{avg_t:.1f}"])

        report_lines.append(format_table(headers, rows))
        report_lines.append(f"\nKey Parameters: key_size={key_size}, "
                             f"locality_size={locality_size}, "
                             f"iterations={n_iterations}")
        report_lines.append(f"Benchmarks processed: {len(good)} success, "
                             f"{len(all_results)-len(good)} error")

        # Detailed table
        report_lines.append("\nDetailed Gate Error Breakdown:")
        headers2 = ['Benchmark', 'GE=0', 'GE=1', 'GE=2', 'N_test', 'ChangePred']
        rows2 = []
        for r in good:
            n = r['test_samples']
            rows2.append([
                r['benchmark'],
                f"{r['gate_error_0']}({r['gate_error_0']/max(n,1)*100:.0f}%)",
                f"{r['gate_error_1']}({r['gate_error_1']/max(n,1)*100:.0f}%)",
                f"{r['gate_error_2']}({r['gate_error_2']/max(n,1)*100:.0f}%)",
                n,
                f"{r['n_predicted_changed']}/{n}"
            ])
        report_lines.append(format_table(headers2, rows2))
    else:
        report_lines.append("No benchmarks completed successfully.")

    report = '\n'.join(report_lines)
    print('\n' + report)

    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\n[*] Report saved to: {output_file}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  Locked-netlist multi-benchmark driver
# ─────────────────────────────────────────────────────────────────────────────

def run_locked_benchmarks(bench_files: List[str],
                          key_files: Dict[str, str] = None,
                          extra_key_size: int = 16,
                          locality_size: int = 6,
                          n_iterations: int = 10,
                          output_file: str = 'sail_locked_results.txt',
                          verbose: bool = False) -> List[Dict]:
    key_files = key_files or {}
    all_results = []
    for bf in bench_files:
        kf = key_files.get(bf)
        print(f"\n[*] Attacking locked netlist: {os.path.basename(bf)}")
        result = attack_locked_benchmark(bf, key_file=kf,
                                          extra_key_size=extra_key_size,
                                          locality_size=locality_size,
                                          n_iterations=n_iterations,
                                          verbose=verbose)
        all_results.append(result)
        if 'error' in result:
            print(f"    ERROR: {result['error']}")
        else:
            acc_str = (f"{result['key_accuracy_pct']:.1f}%" if result['key_accuracy_pct'] is not None
                       else "n/a (no ground truth)")
            print(f"    Real keys: {result['n_real_keys']} | Key accuracy: {acc_str} | "
                  f"Validation Recovery: {result['complete_recovery_pct']:.1f}% | "
                  f"Time: {result['elapsed_sec']:.1f}s")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SAIL Locked-Netlist Attack Results")
    report_lines.append("=" * 70)

    good = [r for r in all_results if 'error' not in r]
    if good:
        headers = ['Benchmark', 'Nodes', 'RealKeys', 'ValRecov%', 'ValR-Metric', 'KeyAcc%', 'Time(s)']
        rows = []
        for r in good:
            rows.append([
                r['benchmark'], r['nodes'], r['n_real_keys'],
                f"{r['complete_recovery_pct']:.1f}", f"{r['r_metric']:.1f}",
                'n/a' if r['key_accuracy_pct'] is None else f"{r['key_accuracy_pct']:.1f}",
                f"{r['elapsed_sec']:.1f}"
            ])
        report_lines.append(format_table(headers, rows))
        report_lines.append(f"\nValRecov%/ValR-Metric are measured on the held-out pseudo-self-reference "
                             f"split (no ground truth exists for the real key gates unless --key_file is given).")

        for r in good:
            report_lines.append(f"\n--- {r['benchmark']}: recovered key-gate localities ---")
            headers2 = ['Key Input', 'Target Gate', 'Raw Type', 'Recovered Type', 'Inferred Key', 'Actual Key']
            rows2 = [[row['key'], row['center'], row['raw_type'], row['recovered_type'],
                      '?' if row['inferred_key'] is None else row['inferred_key'],
                      '-' if row['actual_key'] is None else row['actual_key']]
                     for row in r['key_rows']]
            report_lines.append(format_table(headers2, rows2))
    else:
        report_lines.append("No locked benchmarks completed successfully.")

    report = '\n'.join(report_lines)
    print('\n' + report)

    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\n[*] Report saved to: {output_file}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()
    parser = argparse.ArgumentParser(description='SAIL: ML-Guided Structural Attack on Hardware Obfuscation')
    parser.add_argument('--bench',        type=str, help='Single clean .bench file (SAIL simulates its own obfuscation)')
    parser.add_argument('--bench_dir',    type=str, help='Directory of clean .bench files')
    parser.add_argument('--locked_bench', type=str, help='Single already-locked .bench file to attack directly')
    parser.add_argument('--locked_bench_dir', type=str, help='Directory of already-locked .bench files to attack')
    parser.add_argument('--key_file',     type=str, help='Ground-truth key file (name = 0/1) to score '
                                                           '--locked_bench results; only used for evaluation')
    parser.add_argument('--key_size',     type=int, default=32,  help='Key size for --bench/--bench_dir mode (default: 32)')
    parser.add_argument('--extra_key_size', type=int, default=16, help='Synthetic keys/iteration for the '
                                                           'pseudo-self-reference training set in --locked_bench mode (default: 16)')
    parser.add_argument('--locality_size',type=int, default=6,   help='Locality size (default: 6)')
    parser.add_argument('--iterations',   type=int, default=10,  help='PSR iterations (default: 10)')
    parser.add_argument('--output',       type=str, default=None)
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.locked_bench or args.locked_bench_dir:
        locked_files = []
        if args.locked_bench:
            if not os.path.isfile(args.locked_bench):
                print(f"ERROR: File not found: {args.locked_bench}")
                sys.exit(1)
            locked_files.append(args.locked_bench)
        else:
            locked_files = sorted(glob.glob(os.path.join(args.locked_bench_dir, '*.bench')))
            if not locked_files:
                print(f"ERROR: No .bench files found in {args.locked_bench_dir}")
                sys.exit(1)
            print(f"Found {len(locked_files)} locked benchmark files: "
                  f"{[os.path.basename(f) for f in locked_files]}")

        key_files = {}
        if args.key_file:
            if not os.path.isfile(args.key_file):
                print(f"ERROR: Key file not found: {args.key_file}")
                sys.exit(1)
            if len(locked_files) > 1:
                print("WARNING: --key_file only applies to a single --locked_bench file; ignoring in batch mode")
            else:
                key_files[locked_files[0]] = args.key_file

        run_locked_benchmarks(
            locked_files,
            key_files=key_files,
            extra_key_size=args.extra_key_size,
            locality_size=args.locality_size,
            n_iterations=args.iterations,
            output_file=args.output or 'sail_locked_results.txt',
            verbose=args.verbose
        )
        return

    bench_files = []
    if args.bench:
        if not os.path.isfile(args.bench):
            print(f"ERROR: File not found: {args.bench}")
            sys.exit(1)
        bench_files.append(args.bench)
    elif args.bench_dir:
        bench_files = sorted(glob.glob(os.path.join(args.bench_dir, '*.bench')))
        if not bench_files:
            bench_files = sorted(glob.glob(os.path.join(args.bench_dir, '**', '*.bench'),
                                            recursive=True))
        if not bench_files:
            print(f"ERROR: No .bench files found in {args.bench_dir}")
            sys.exit(1)
        print(f"Found {len(bench_files)} benchmark files: "
              f"{[os.path.basename(f) for f in bench_files]}")
    else:
        parser.print_help()
        print("\nERROR: Please provide --bench/--bench_dir (simulate obfuscation) "
              "or --locked_bench/--locked_bench_dir (attack a real locked netlist)")
        sys.exit(1)

    run_benchmarks(
        bench_files,
        key_size=args.key_size,
        locality_size=args.locality_size,
        n_iterations=args.iterations,
        output_file=args.output or 'sail_results.txt',
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()