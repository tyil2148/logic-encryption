"""
Usage
-----
  python ttlock.py  <input.bench>  <output_locked.bench>  [--key-size N] [--seed S]

  --key-size  N   Number of key bits (default: 16)
  --seed      S   Random seed for reproducible key / protected pattern selection.
"""

import argparse
import random
import sys
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


# Holds the parsed netlist of a .bench file.
class BenchCircuit:
    def __init__(self):
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        # gate_def[wire] = (gate_type, [input_wires])
        self.gate_def: Dict[str, Tuple[str, List[str]]] = {}
        # wires that are primary inputs (no gate driving them)
        self.pi_set: Set[str] = set()
        self.po_set: Set[str] = set()

    def parse(self, path: str) -> "BenchCircuit":
        with open(path, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                up = line.upper()

                if up.startswith("INPUT("):
                    name = _extract_single(line)
                    self.inputs.append(name)
                    self.pi_set.add(name)

                elif up.startswith("OUTPUT("):
                    name = _extract_single(line)
                    self.outputs.append(name)
                    self.po_set.add(name)

                elif "=" in line:
                    lhs, rhs = line.split("=", 1)
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    paren = rhs.index("(")
                    gate_type = rhs[:paren].strip().upper()
                    args_str = rhs[paren + 1 : rhs.rindex(")")].strip()
                    args = [a.strip() for a in args_str.split(",") if a.strip()]
                    self.gate_def[lhs] = (gate_type, args)

        return self

    def write(self, path: str):
        with open(path, "w") as fh:
            fh.write("# TTLock-protected netlist\n")
            for pi in self.inputs:
                fh.write(f"INPUT({pi})\n")
            fh.write("\n")
            for po in self.outputs:
                fh.write(f"OUTPUT({po})\n")
            fh.write("\n")
            for wire, (gtype, fans) in self.gate_def.items():
                fans_str = ", ".join(fans)
                fh.write(f"{wire} = {gtype}({fans_str})\n")

        print(f"[TTLock] Locked netlist written to: {path}")

# Helper functions
#extract the gate name
def _extract_single(line: str) -> str:
    return line[line.index("(") + 1 : line.rindex(")")].strip()

#Return a wire name from a fresh base
def _fresh(base: str, existing: Set[str]) -> str:
    name = base
    idx = 0
    while name in existing:
        name = f"{base}_{idx}"
        idx += 1
    existing.add(name)
    return name

def _all_wires(circ: BenchCircuit) -> Set[str]:
    wires: Set[str] = set(circ.pi_set)
    wires.update(circ.gate_def.keys())
    return wires

# logic cone identification

# does a BFS/DPS backward from the output wire and returns all the wires within its fanin cone
def _fanin_cone(output_wire: str, circ: BenchCircuit) -> Set[str]:
    cone: Set[str] = set()
    queue = deque([output_wire])
    while queue:
        w = queue.popleft()
        if w in cone:
            continue
        cone.add(w)
        if w in circ.gate_def:
            _, fans = circ.gate_def[w]
            queue.extend(fans)
    return cone

#pick a primary output to protect. need a primary output that is driven by a non-trivial cone
def pick_target_output(circ: BenchCircuit, rng: random.Random) -> str:
    candidates = []
    for po in circ.outputs:
        cone = _fanin_cone(po, circ)
        gate_wires = [w for w in cone if w in circ.gate_def]
        if len(gate_wires) >= 4:
            candidates.append(po)
    if not candidates:
        candidates = circ.outputs[:]
    return rng.choice(candidates)

# returns the primary inputs that feed into the cone
def cone_primary_inputs(cone: Set[str], circ: BenchCircuit) -> List[str]:
    return [w for w in cone if w in circ.pi_set]

"""
1. Choose a target primary output (PO).
2. Select n key bits = n primary inputs feeding the PO's fanin cone
    (padded with fresh key inputs if the cone has fewer than n PIs).
3. Generate a random secret key K* ∈ {0,1}^n and protected pattern P* = K*.
4. Build the Modified Logic Cone (MLC):
        – Identify the gate that drives the PO.
        – Insert an inversion trigger: an n-input AND tree that fires only
        when all inputs match P*.  XOR this with the PO gate output to
        flip the output for exactly pattern P*.
5. Build the Restore Unit:
        – n-bit comparator: for each key bit k_i, emit an XNOR(pi_i, key_i_wire).
        AND all XNOR outputs → comparator_out (= 1 iff IN == K*).
        – Final XOR(mlc_out, comparator_out) = correct output when key is right,
        still wrong otherwise.
6. Route the PO through the final XOR.
7. Add key inputs (KEY_0 … KEY_{n-1}) as primary inputs with their correct
    values written to a separate .key file.
"""
class TTLock:
    def __init__(self, circ: BenchCircuit, key_size: int, rng: random.Random):
        self.circ = circ
        self.key_size = key_size
        self.rng = rng
        self.existing_wires = _all_wires(circ)
        self.key_bits: List[int] = []          # secret key values
        self.protected_pattern: List[int] = []  # = key_bits (P* == K*)
        self.target_po: str = ""
        self.key_input_wires: List[str] = []   # KEY_0, KEY_1, …
        self.cone_pi_wires: List[str] = []     # PIs assigned to key positions

    def lock(self) -> BenchCircuit:
        circ = self.circ

        # choose target primary output
        self.target_po = pick_target_output(circ, self.rng)
        print(f"[TTLock] Target output      : {self.target_po}")

        # select n amount of cones for the fanin
        cone = _fanin_cone(self.target_po, circ)
        cone_pis = cone_primary_inputs(cone, circ)
        self.rng.shuffle(cone_pis)
        n = self.key_size
        if len(cone_pis) >= n:
            selected_pis = cone_pis[:n]
        else:
            # use all available cone PIs, then supplement with other circuit PIs
            other_pis = [p for p in circ.inputs if p not in cone_pis]
            self.rng.shuffle(other_pis)
            selected_pis = cone_pis + other_pis[: n - len(cone_pis)]
            # if still not enough, pad with fresh synthetic inputs
            while len(selected_pis) < n:
                new_pi = _fresh(f"ttlock_pi_{len(selected_pis)}", self.existing_wires)
                circ.inputs.append(new_pi)
                circ.pi_set.add(new_pi)
                selected_pis.append(new_pi)

        self.cone_pi_wires = selected_pis
        print(f"[TTLock] Cone PIs used      : {len(selected_pis)}  (key-size = {n})")

        # generate secret key and protected pattern (P* = K*)
        self.key_bits = [self.rng.randint(0, 1) for _ in range(n)]
        self.protected_pattern = self.key_bits[:]
        print(f"[TTLock] Secret key (K*)    : {''.join(map(str, self.key_bits))}")

        # create key input wires
        self.key_input_wires = []
        for i in range(n):
            kw = _fresh(f"keyinput{i}", self.existing_wires)
            self.key_input_wires.append(kw)
            circ.inputs.append(kw)
            circ.pi_set.add(kw)
        """
        We want a signal that is 1 iff all selected_pis match the protected pattern.
        For each pi, if pattern bit == 1 → use pi directly; if 0 → invert pi.
        AND everything together → pattern_detect signal.
        XOR(original_po_driver, pattern_detect) → mlc_out
        This inverts the PO for exactly the protected pattern.
        """
        mlc_out = self._build_mlc(selected_pis, self.protected_pattern)

        """
        Comparator: XNOR(pi_i, key_i) for each i, then AND all → cmp_out
        cmp_out == 1 iff IN == K* (correct key).
        Final XOR(mlc_out, cmp_out) restores correctness when key matches.
        """
        restore_out = self._build_restore(selected_pis, self.key_input_wires)

        # connect final XOR to PO
        final_xor = _fresh(f"ttlock_final_xor", self.existing_wires)
        circ.gate_def[final_xor] = ("XOR", [mlc_out, restore_out])

        # Redirect the PO to the final_xor output
        # In .bench, POs are named in OUTPUT() declarations
        # must produce a wire with the same name (or we rename via BUF).
        po_name = self.target_po

        if po_name in circ.gate_def:
            # rename the original PO gate output so we can reuse the name
            old_driver = _fresh(f"{po_name}_orig", self.existing_wires)
            circ.gate_def[old_driver] = circ.gate_def.pop(po_name)
            # Patch all references to po_name in other gates
            for w, (gt, fans) in circ.gate_def.items():
                circ.gate_def[w] = (gt, [old_driver if f == po_name else f for f in fans])
            # Now re-insert the mlc and restore logic using old_driver as base
            # (already done above with mlc_out / restore_out referencing old_driver)
            # Wire the final XOR to the PO name
            circ.gate_def[po_name] = ("BUF", [final_xor])
        else:
            # PO is directly a PI (unlikely but handle it)
            circ.gate_def[po_name] = ("BUF", [final_xor])

        print(f"[TTLock] Locking complete.  Added {2*n + 3} logic nodes.")
        return circ

    """
    Build the Modified Logic Cone output:
        pattern_detect AND-tree → XOR with PO driver.
    Returns the wire name of the mlc output.
    """
    def _build_mlc(self, pis: List[str], pattern: List[int]) -> str:
        circ = self.circ
        # build per-bit match wires: if pattern[i]==0 → NOT(pi), else pi directly
        bit_wires = []
        for i, (pi, bit) in enumerate(zip(pis, pattern)):
            if bit == 0:
                inv_w = _fresh(f"ttlock_mlc_inv{i}", self.existing_wires)
                circ.gate_def[inv_w] = ("NOT", [pi])
                bit_wires.append(inv_w)
            else:
                bit_wires.append(pi)

        # AND-tree over bit_wires → pattern_detect
        detect = self._and_tree(bit_wires, prefix="ttlock_mlc_and")

        # XOR with original PO driver
        # The original PO driver wire is the wire named target_po at this point
        # (before we rename it in lock()).  We reference it symbolically; the
        # caller (lock) will rename the original gate after this call returns.
        # So we use self.target_po as the fanin here — lock() will rename it.
        xor_w = _fresh("ttlock_mlc_xor", self.existing_wires)
        circ.gate_def[xor_w] = ("XOR", [self.target_po, detect])
        return xor_w

    """
    Build the Restore Unit:
        XNOR(pi_i, key_i) for each i → AND-tree → cmp_out
    Returns the wire name of the comparator output.
    """
    def _build_restore(self, pis: List[str], key_wires: List[str]) -> str:
        circ = self.circ
        xnor_wires = []
        for i, (pi, kw) in enumerate(zip(pis, key_wires)):
            xnor_w = _fresh(f"ttlock_xnor{i}", self.existing_wires)
            circ.gate_def[xnor_w] = ("XNOR", [pi, kw])
            xnor_wires.append(xnor_w)
        return self._and_tree(xnor_wires, prefix="ttlock_cmp_and")

# Reduce a list of wires with a balanced 'and' tree- returns output wire
    def _and_tree(self, wires: List[str], prefix: str) -> str:
        circ = self.circ
        if len(wires) == 1:
            return wires[0]
        level = wires[:]
        stage = 0
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level) - 1, 2):
                and_w = _fresh(f"{prefix}_s{stage}_g{i//2}", self.existing_wires)
                circ.gate_def[and_w] = ("AND", [level[i], level[i + 1]])
                next_level.append(and_w)
            if len(level) % 2 == 1:
                next_level.append(level[-1])
            level = next_level
            stage += 1
        return level[0]

#write the secret key into a .key file
    def write_key(self, path: str):
        with open(path, "w") as fh:
            fh.write("# TTLock Secret Key File\n")
            fh.write(f"# Target output : {self.target_po}\n")
            fh.write(f"# Key size      : {self.key_size}\n\n")
            for kw, kv in zip(self.key_input_wires, self.key_bits):
                fh.write(f"{kw} = {kv}\n")
        print(f"[TTLock] Secret key written to    : {path}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  TTLock Locking Summary")
        print("=" * 60)
        print(f"  Protected output   : {self.target_po}")
        print(f"  Key size (n)       : {self.key_size}")
        print(f"  Secret key K*      : {''.join(map(str, self.key_bits))}")
        print(f"  Protected pattern  : {''.join(map(str, self.protected_pattern))}")
        print(f"  SAT attack DIPs    : ~2^{self.key_size} - 1  "
              f"≈ {2**self.key_size - 1:,}")
        print(f"  Key inputs added   : {', '.join(self.key_input_wires[:4])}"
              f"{'...' if self.key_size > 4 else ''}")
        print("=" * 60 + "\n")

#sanity checks on netlist
def validate_bench(circ: BenchCircuit) -> bool:
    ok = True
    all_defined = set(circ.pi_set) | set(circ.gate_def.keys())
    for wire, (gtype, fans) in circ.gate_def.items():
        for f in fans:
            if f not in all_defined:
                print(f"[WARN] Undefined fanin '{f}' in gate '{wire}'")
                ok = False
    for po in circ.outputs:
        if po not in all_defined:
            print(f"[WARN] Output '{po}' has no driver")
            ok = False
    return ok

def circuit_stats(circ: BenchCircuit, label: str = ""):
    tag = f"[{label}] " if label else ""
    gate_counts: Dict[str, int] = defaultdict(int)
    for _, (gt, _) in circ.gate_def.items():
        gate_counts[gt] += 1

    print(f"\n{tag}Circuit statistics:")
    print(f"  Primary inputs  : {len(circ.inputs)}")
    print(f"  Primary outputs : {len(circ.outputs)}")
    print(f"  Total gates     : {len(circ.gate_def)}")
    for gt, cnt in sorted(gate_counts.items()):
        print(f"    {gt:8s} : {cnt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTLock Logic Locking ")
    p.add_argument("input",  help="Input .bench file")
    p.add_argument("output", help="Output locked .bench file")
    p.add_argument(
        "--key-size", "-n", type=int, default=16,
        help="Number of key bits (default: 16).  "
             "SAT-attack complexity ≈ 2^n iterations."
    )
    p.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility (default: random)"
    )
    p.add_argument(
        "--key-file", "-k", type=str, default=None,
        help="Path to write the secret key (default: <output>.key)"
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run basic netlist validation before and after locking"
    )
    return p.parse_args()

def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    rng  = random.Random(seed)
    print(f"[TTLock] Random seed: {seed}")

    print(f"[TTLock] Parsing: {args.input}")
    circ = BenchCircuit().parse(args.input)
    circuit_stats(circ, "Original")

    if args.validate:
        ok = validate_bench(circ)
        print(f"[TTLock] Original netlist valid: {ok}")

    if not circ.outputs:
        print("[ERROR] No primary outputs found in the circuit.")
        sys.exit(1)

    locker = TTLock(circ, key_size=args.key_size, rng=rng)
    locked = locker.lock()

    if args.validate:
        ok = validate_bench(locked)
        print(f"[TTLock] Locked  netlist valid: {ok}")

    locker.print_summary()
    circuit_stats(locked, "Locked")
    locked.write(args.output)
    key_path = args.key_file or (args.output.replace(".bench", "") + ".key")
    locker.write_key(key_path)

    print("\n[TTLock] Done.")
if __name__ == "__main__":
    main()