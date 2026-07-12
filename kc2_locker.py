"""
lock_circuit.py — Random XOR/XNOR Logic Locking
=================================================
Implements the exact locking scheme used in the KC2 paper (DATE 2019):

  "We used random XOR/XNOR locking [18] to obfuscate the designs with
   various overhead values."
  [18] Subramanyan et al., "Evaluating the security of logic encryption
       algorithms," HOST 2015.

How it works:
  1. Parse the input .bench netlist
  2. Identify all lockable signals — internal gate outputs that are not
     primary inputs, key inputs, or circuit outputs (we can insert on
     any internal wire)
  3. Randomly select N signals to lock (N determined by --overhead or --keys)
  4. For each selected signal S:
       - Generate a new key bit k_i
       - Generate a new gate name: lck_S
       - Randomly choose XOR or XNOR
       - Insert: lck_S = XOR(S, k_i)   [or XNOR]
       - Rewrite all fanout of S (except the gate that produces S) to use lck_S
       - The correct key value k_i* is:
           0 for XOR  (XOR with 0 = passthrough)
           1 for XNOR (XNOR with 1 = passthrough)
  5. Write the locked .bench file with KEY() directives
  6. Write a .key file recording the correct key

The oracle is the ORIGINAL unlocked file — KC2 queries it directly.

Usage:
  # Lock with a specific number of key bits
  python lock_circuit.py --input s27.bench --keys 8 --output s27_locked.bench

  # Lock with a gate-count overhead percentage (matches paper's methodology)
  python lock_circuit.py --input s27.bench --overhead 10 --output s27_locked.bench

  # Lock with a fixed seed for reproducibility
  python lock_circuit.py --input s27.bench --keys 8 --seed 42 --output s27_locked.bench

  # Then attack with KC2:
  python kc2.py --locked s27_locked.bench --oracle s27.bench

Output files:
  s27_locked.bench  — the locked netlist with KEY() inputs
  s27_locked.key    — the correct key values (ground truth)
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# PARSER — reuse the same .bench format as kc2.py
# =============================================================================

@dataclass
class Gate:
    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_dff: bool = False
    is_key: bool = False


@dataclass
class Circuit:
    gates: Dict[str, Gate] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dffs: List[str] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    # Preserve original line order for faithful reconstruction
    line_order: List[str] = field(default_factory=list)


def parse_bench(text: str) -> Circuit:
    """Parse a .bench netlist. Handles both port-style and assignment-style DFFs."""
    c = Circuit()

    def strip_comment(line: str) -> str:
        if '#' in line:
            line = line[:line.index('#')]
        return line.strip()

    lines = [strip_comment(l) for l in text.splitlines()]
    lines = [l for l in lines if l]

    dff_pairs: List[Tuple[str, str]] = []

    for line in lines:
        up = line.upper()
        if up.startswith("INPUT("):
            name = line[6:-1].strip()
            g = Gate(name=name, op="INPUT", is_input=True)
            c.gates[name] = g
            c.inputs.append(name)
        elif up.startswith("KEY("):
            name = line[4:-1].strip()
            g = Gate(name=name, op="KEY", is_key=True, is_input=True)
            c.gates[name] = g
            c.keys.append(name)
        elif up.startswith("OUTPUT("):
            name = line[7:-1].strip()
            c.outputs.append(name)
            if name in c.gates:
                c.gates[name].is_output = True
        elif up.startswith("DFF("):
            # Port-style: DFF(D_input, Q_output)
            inner = line[4:-1]
            parts = [p.strip() for p in inner.split(",")]
            dff_pairs.append((parts[0], parts[1]))
        elif "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if "(" not in rhs:
                # Wire alias → BUF
                g = Gate(name=lhs, op="BUF", inputs=[rhs])
                c.gates[lhs] = g
            else:
                paren = rhs.index("(")
                op = rhs[:paren].strip().upper()
                ins_str = rhs[paren+1:-1]
                ins = [i.strip() for i in ins_str.split(",") if i.strip()]
                if op == "DFF":
                    # Assignment-style DFF: Q = DFF(D)
                    g = Gate(name=lhs, op="DFF", inputs=ins, is_dff=True)
                    c.gates[lhs] = g
                    if lhs not in c.dffs:
                        c.dffs.append(lhs)
                else:
                    g = Gate(name=lhs, op=op, inputs=ins)
                    c.gates[lhs] = g

    # Register port-style DFFs
    for d_input, q_output in dff_pairs:
        if q_output not in c.gates:
            g = Gate(name=q_output, op="DFF", inputs=[d_input], is_dff=True)
            c.gates[q_output] = g
        if q_output not in c.dffs:
            c.dffs.append(q_output)

    # Mark outputs
    for o in c.outputs:
        if o in c.gates:
            c.gates[o].is_output = True

    return c


# =============================================================================
# LOCKING — random XOR/XNOR insertion
# =============================================================================

def count_gates(circuit: Circuit) -> int:
    """
    Count the number of logic gates (excluding inputs, keys, DFFs, wire aliases).
    This matches the paper's gate-count metric.
    """
    non_gates = {"INPUT", "KEY", "DFF", "BUF", "BUFF"}
    return sum(1 for g in circuit.gates.values()
               if g.op not in non_gates and not g.is_input and not g.is_dff)


def find_lockable_signals(circuit: Circuit) -> List[str]:
    """
    Find all internal signals that can have an XOR/XNOR key gate inserted.

    We only lock INTERNAL signals (not primary outputs) so that the locked
    circuit's output signal names always match the oracle's output names.
    This is required for the oracle comparison in KC2 to work correctly.

    A signal is lockable if:
      - It is the output of a logic gate (not an input, key, or DFF Q output)
      - It is NOT a primary circuit output (preserving output names)
      - It has at least one fanout consumer (another gate reads it)
    """
    all_inputs_to_gates: Set[str] = set()
    for g in circuit.gates.values():
        for inp in g.inputs:
            all_inputs_to_gates.add(inp)

    primary_outputs: Set[str] = set(circuit.outputs)
    excluded_ops = {"INPUT", "KEY", "DFF"}

    lockable = []
    for name, g in circuit.gates.items():
        if g.op in excluded_ops or g.is_input or g.is_dff:
            continue
        if name in primary_outputs:
            continue  # skip outputs — preserve their names for oracle matching
        if name in all_inputs_to_gates:
            lockable.append(name)

    return lockable


def lock_circuit(
    circuit: Circuit,
    n_keys: int,
    seed: Optional[int] = None,
) -> Tuple[Circuit, Dict[str, int]]:
    """
    Apply random XOR/XNOR locking to a circuit.

    For each of the n_keys key bits:
      1. Randomly choose a lockable signal S
      2. Randomly choose XOR or XNOR
      3. Insert a new gate:  lck_<S> = XOR(S, k_i)
      4. Redirect all consumers of S to use lck_<S> instead
      5. Record the correct key value:
           XOR  → k* = 0 (XOR with 0 is a passthrough)
           XNOR → k* = 1 (XNOR with 1 is a passthrough)

    Returns:
      locked_circuit : the modified circuit with KEY() inputs added
      correct_key    : dict mapping key_name → correct_value
    """
    rng = random.Random(seed)

    lockable = find_lockable_signals(circuit)
    if n_keys > len(lockable):
        print(f"[WARNING] Requested {n_keys} keys but only {len(lockable)} "
              f"lockable signals exist. Locking all {len(lockable)}.")
        n_keys = len(lockable)

    # Choose which signals to lock (without replacement — each signal locked once)
    chosen_signals = rng.sample(lockable, n_keys)

    # Deep copy the circuit structure
    import copy
    locked = copy.deepcopy(circuit)

    correct_key: Dict[str, int] = {}

    for i, signal in enumerate(chosen_signals):
        key_name = f"key_{i}"
        lock_gate_name = f"lck_{signal}_{i}"

        # Choose XOR or XNOR randomly
        use_xnor = rng.choice([False, True])
        op = "XNOR" if use_xnor else "XOR"

        # Correct key value:
        #   XOR  with k=0 → output = signal (unchanged)  → k* = 0
        #   XNOR with k=1 → output = signal (unchanged)  → k* = 1
        correct_value = 1 if use_xnor else 0
        correct_key[key_name] = correct_value

        # Add key input gate
        key_gate = Gate(name=key_name, op="KEY", is_key=True, is_input=True)
        locked.gates[key_name] = key_gate
        locked.keys.append(key_name)

        # Add locking gate: lck_signal_i = XOR(signal, key_i)
        lock_gate = Gate(name=lock_gate_name, op=op, inputs=[signal, key_name])
        locked.gates[lock_gate_name] = lock_gate

        # Redirect all consumers of `signal` to use `lock_gate_name` instead.
        # Since find_lockable_signals() excludes primary outputs, `signal` is
        # always an internal wire here — simple fanout redirection suffices.
        for gname, g in locked.gates.items():
            if gname == lock_gate_name:
                continue  # don't redirect the lock gate itself
            if gname == signal:
                continue  # don't redirect the gate that produces signal
            g.inputs = [
                lock_gate_name if inp == signal else inp
                for inp in g.inputs
            ]
        lock_gate.is_output = False

    return locked, correct_key


# =============================================================================
# WRITER — output the locked .bench file
# =============================================================================

def write_bench(circuit: Circuit, path: str, header_comment: str = "") -> None:
    """
    Write a Circuit object back to .bench format.

    Output order:
      1. Header comment
      2. INPUT() declarations (non-key primary inputs)
      3. KEY() declarations
      4. OUTPUT() declarations
      5. DFF declarations (assignment style: Q = DFF(D))
      6. Logic gate assignments
    """
    lines = []

    if header_comment:
        for line in header_comment.splitlines():
            lines.append(f"# {line}")
        lines.append("")

    # Primary inputs
    for name in circuit.inputs:
        lines.append(f"INPUT({name})")
    lines.append("")

    # Key inputs
    for name in circuit.keys:
        lines.append(f"KEY({name})")
    lines.append("")

    # Outputs
    for name in circuit.outputs:
        lines.append(f"OUTPUT({name})")
    lines.append("")

    # DFFs first (they're sources, readers need to see them defined)
    for name in circuit.dffs:
        g = circuit.gates[name]
        d_input = g.inputs[0]
        lines.append(f"{name} = DFF({d_input})")
    if circuit.dffs:
        lines.append("")

    # All other gates (excluding inputs, keys, outputs-that-are-inputs, DFFs)
    skip_ops = {"INPUT", "KEY", "DFF"}
    for name, g in circuit.gates.items():
        if g.op in skip_ops or g.is_input or g.is_dff:
            continue
        ins_str = ", ".join(g.inputs)
        lines.append(f"{name} = {g.op}({ins_str})")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_key_file(correct_key: Dict[str, int], path: str,
                   locked_file: str, oracle_file: str) -> None:
    """Write the correct key to a .key file."""
    lines = [
        f"# Correct key for: {locked_file}",
        f"# Oracle (original unlocked): {oracle_file}",
        f"# Format: key_name = correct_value",
        f"# Pass to KC2 with: --true-key " +
        ",".join(f"{k}={v}" for k, v in correct_key.items()),
        "",
    ]
    for key_name, value in correct_key.items():
        lines.append(f"{key_name} = {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# VERIFICATION — sanity check the locked circuit
# =============================================================================

def verify_locking(
    original: Circuit,
    locked: Circuit,
    correct_key: Dict[str, int],
    n_trials: int = 500,
    seed: int = 0,
) -> Tuple[bool, float]:
    """
    Verify that the locked circuit with the correct key produces the same
    outputs as the original circuit on random input sequences.

    Returns (all_passed, accuracy).
    """
    import itertools

    rng = random.Random(seed)

    def sim_step(circuit, pi, key, state):
        """Single-cycle simulation."""
        vals = {}
        vals.update(pi)
        vals.update(key)
        vals.update(state)

        # Build eval plan if not cached
        if not hasattr(circuit, '_vplan'):
            from collections import deque, defaultdict
            sources = {n for n, g in circuit.gates.items()
                       if g.op in ("INPUT", "KEY") or g.is_dff}
            in_deg = defaultdict(int)
            children = defaultdict(list)
            for gname, g in circuit.gates.items():
                if gname in sources: continue
                for pin in g.inputs:
                    if pin not in sources:
                        children[pin].append(gname)
                        in_deg[gname] += 1
            q = deque(n for n in circuit.gates if n not in sources and in_deg[n] == 0)
            order = []
            while q:
                node = q.popleft()
                order.append(node)
                for child in children[node]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        q.append(child)
            circuit._vplan = [
                (n, circuit.gates[n].op, circuit.gates[n].inputs)
                for n in order
                if not (circuit.gates[n].op in ("INPUT","KEY") or circuit.gates[n].is_dff)
            ]
            circuit._dff_plan = [(q, circuit.gates[q].inputs[0]) for q in circuit.dffs]

        for gname, op, inputs in circuit._vplan:
            ins = [vals[i] for i in inputs]
            if op == "AND":         vals[gname] = int(all(ins))
            elif op == "OR":        vals[gname] = int(any(ins))
            elif op == "NAND":      vals[gname] = int(not all(ins))
            elif op == "NOR":       vals[gname] = int(not any(ins))
            elif op == "XOR":       vals[gname] = ins[0] ^ ins[1]
            elif op == "XNOR":      vals[gname] = int(not (ins[0] ^ ins[1]))
            elif op == "NOT":       vals[gname] = 1 - ins[0]
            elif op in ("BUF","BUFF"): vals[gname] = ins[0]

        next_state = {q: vals.get(d, 0) for q, d in circuit._dff_plan}
        outputs = {o: vals[o] for o in circuit.outputs}
        return outputs, next_state

    correct = 0
    total = 0

    for _ in range(n_trials):
        seq_len = rng.randint(3, 10)
        inp_seq = [
            {i: rng.randint(0, 1) for i in original.inputs}
            for _ in range(seq_len)
        ]

        orig_state = {q: 0 for q in original.dffs}
        lock_state = {q: 0 for q in locked.dffs}

        for pi in inp_seq:
            orig_out, orig_state = sim_step(original, pi, {}, orig_state)
            lock_out, lock_state = sim_step(locked, pi, correct_key, lock_state)

            # Compare outputs — map by position if names differ
            orig_vals = [v for _, v in sorted(orig_out.items())]
            lock_vals = [v for _, v in sorted(lock_out.items())]

            total += 1
            if orig_vals == lock_vals:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy == 1.0, accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Random XOR/XNOR Logic Locking — matches KC2 paper methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lock s27.bench with 4 key bits (suitable for quick KC2 test)
  python lock_circuit.py --input s27.bench --keys 4

  # Lock at 10% gate-count overhead (matches the paper's Table II)
  python lock_circuit.py --input s27.bench --overhead 10

  # Reproducible locking with fixed seed
  python lock_circuit.py --input s27.bench --keys 8 --seed 42

  # Then attack with KC2 (oracle is the original unlocked file):
  python kc2.py --locked s27_locked.bench --oracle s27.bench

Key sizes from the paper's experiments:
  The paper tests 1%, 5%, and 10% overhead on ISCAS89 circuits.
  For s27 (13 gates), that's approximately:
    1%  overhead → 1 key bit  (too few — use at least 4)
    5%  overhead → 1 key bit
    10% overhead → 2 key bits
  For larger circuits like s35932 (17793 gates):
    1%  overhead → 177 key bits
    5%  overhead → 889 key bits
    10% overhead → 1779 key bits
        """
    )
    parser.add_argument("--input", required=True,
                        help="Input .bench file (unlocked circuit)")
    parser.add_argument("--output", default=None,
                        help="Output locked .bench file "
                             "(default: <input>_locked.bench)")
    parser.add_argument("--keys", type=int, default=None,
                        help="Number of key bits to insert")
    parser.add_argument("--overhead", type=float, default=None,
                        help="Gate-count overhead percentage (e.g. 10 for 10%%)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification after locking")
    parser.add_argument("--xor-only", action="store_true",
                        help="Use only XOR gates (no XNOR). All correct key bits = 0.")

    args = parser.parse_args()

    if args.keys is None and args.overhead is None:
        parser.error("Specify either --keys or --overhead")
    if args.keys is not None and args.overhead is not None:
        parser.error("Specify either --keys or --overhead, not both")

    # ── Read input ────────────────────────────────────────────────────────────
    try:
        text = open(args.input, 'rb').read().decode('latin-1')
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    circuit = parse_bench(text)
    n_gates = count_gates(circuit)
    lockable = find_lockable_signals(circuit)

    print(f"[INFO] Circuit      : {args.input}")
    print(f"[INFO] Primary inputs: {len(circuit.inputs)}")
    print(f"[INFO] Outputs       : {len(circuit.outputs)}")
    print(f"[INFO] DFFs          : {len(circuit.dffs)}")
    print(f"[INFO] Logic gates   : {n_gates}")
    print(f"[INFO] Lockable sigs : {len(lockable)}")
    print(f"[INFO] Sequential    : {len(circuit.dffs) > 0}")

    # ── Determine number of keys ──────────────────────────────────────────────
    if args.overhead is not None:
        n_keys = max(1, round(n_gates * args.overhead / 100.0))
        print(f"[INFO] Overhead      : {args.overhead}% of {n_gates} gates "
              f"→ {n_keys} key bits")
    else:
        n_keys = args.keys

    print(f"[INFO] Key bits      : {n_keys}")
    print(f"[INFO] Search space  : 2^{n_keys} = {2**n_keys} candidates")
    print(f"[INFO] Random seed   : {args.seed if args.seed is not None else 'random'}")

    if n_keys > 20:
        print(f"[WARNING] {n_keys} key bits = {2**n_keys} candidates. "
              f"KC2 enumerates the full key space — this will be very slow "
              f"without a SAT-based key-space representation. "
              f"Consider using --keys 8 or fewer for testing.")

    # ── Apply locking ─────────────────────────────────────────────────────────
    if args.xor_only:
        # Monkeypatch: force XOR only
        _orig_choice = random.Random.choice
        random.Random.choice = lambda self, seq: False  # always XOR
    
    locked, correct_key = lock_circuit(circuit, n_keys, seed=args.seed)

    if args.xor_only:
        random.Random.choice = _orig_choice

    print(f"\n[INFO] Correct key   : {correct_key}")
    print(f"[INFO] KC2 --true-key: " +
          ",".join(f"{k}={v}" for k, v in correct_key.items()))

    # ── Verify ────────────────────────────────────────────────────────────────
    if not args.no_verify:
        print("\n[INFO] Verifying locked circuit...")
        passed, acc = verify_locking(circuit, locked, correct_key)
        if passed:
            print(f"[INFO] Verification PASSED ✓  (accuracy={acc*100:.1f}%)")
        else:
            print(f"[ERROR] Verification FAILED ✗  (accuracy={acc*100:.1f}%)")
            print("[ERROR] The locked circuit with correct key does not match "
                  "the original. This is a bug — please report it.")
            sys.exit(1)

    # ── Write output ──────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_locked.bench")

    key_path = output_path.with_suffix('.key')

    overhead_str = (f"{args.overhead}%" if args.overhead
                    else f"{n_keys} key bits")
    header = (
        f"XOR/XNOR locked circuit\n"
        f"Original : {args.input}\n"
        f"Overhead : {overhead_str}\n"
        f"Key bits : {n_keys}\n"
        f"Seed     : {args.seed}\n"
        f"Scheme   : random XOR/XNOR (Subramanyan et al. HOST 2015)\n"
        f"\n"
        f"To attack with KC2:\n"
        f"  python kc2.py --locked {output_path.name} "
        f"--oracle {input_path.name}"
    )

    write_bench(locked, str(output_path), header_comment=header)
    write_key_file(correct_key, str(key_path),
                   str(output_path), args.input)

    print(f"\n[INFO] Locked netlist: {output_path}")
    print(f"[INFO] Key file      : {key_path}")
    print(f"\nTo attack:")
    print(f"  python kc2.py --locked {output_path.name} "
          f"--oracle {input_path.name}")


if __name__ == "__main__":
    main()