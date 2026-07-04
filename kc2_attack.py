"""
KC2: Key-Condition Crunching for Fast Sequential Circuit Deobfuscation
=====================================================================

An oracle-guided deobfuscation (SAT) attack on logic-locked .bench circuits,
following Shamsi, Li, Pan and Jin, "KC2: Key-Condition Crunching for Fast
Sequential Circuit Deobfuscation" (DATE 2019).

This recovers the correct key of a circuit locked by the lockers in this repo
(epic.py / sarlock.py / ttlock.py) given oracle access to the original
unlocked design.

Usage
-----
    python kc2_attack.py LOCKED.bench --oracle ORIGINAL.bench

    optional:
      --max-iter N     cap on DIP/DIS queries          (default 100000)
      --max-depth N    cap on sequential unroll depth   (default 512)
      --no-crunch      disable key-condition crunching (run plain baseline)
      --seed N         RNG seed for verification patterns
      --quiet          less logging

Scope / faithfulness
--------------------
This is a pure-Python implementation built on python-sat's Glucose4, matching
the paper's choice of Glucose with *incremental* solving and assumptions.
The KC2 "core" simplification layer is implemented:

  * one persistent incremental SAT instance reused across the whole attack
    (sec. III-B);
  * key-condition crunching by constant-propagating each I/O constraint into
    a compact, cofactored circuit copy over the *shared* key variables, so the
    comparator clauses do not pile up the way the naive attack inflates them
    (sec. III-A/C);
  * negative-key-condition compression: each discriminating query's losing
    candidate key is generalized to a blocking cube by literal dropping
    (sec. III-E);
  * Combinational-Equivalence (CE) termination -- which the paper notes is
    strictly stronger than Unique-Key (UK) -- for the sequential attack
    (sec. III-G).

The heavier NEOS machinery (CUDD BDD/cut sweeping, key-condition->BDD
conversion, and interpolation-based unbounded model checking) is intentionally
NOT reproduced; the paper itself reports UMC "rarely becomes necessary".
"""

import argparse
import random
import sys
import time
from collections import defaultdict, deque

try:
    from pysat.solvers import Glucose4
    from pysat.formula import IDPool
except ImportError:
    sys.exit("This attack requires python-sat. Install with:  pip install python-sat")


# ---------------------------------------------------------------------------
# Bench parsing / circuit model
# ---------------------------------------------------------------------------

_ALIAS = {'BUFF': 'BUF', 'INV': 'NOT', 'NOT1': 'NOT'}
_CONST = {'0', '1', "1'b0", "1'b1"}


class Circuit:
    """A combinational-or-sequential gate-level netlist parsed from .bench.

    Flip-flops (DFF) are treated as state latches: a DFF output wire is a
    "source" (its value is the current state), and the wire feeding the DFF
    is the corresponding next-state.
    """

    def __init__(self, inputs, outputs, gates):
        self.gates = gates                      # name -> {'type', 'inputs'}
        self.dffs = [w for w, g in gates.items() if g['type'] == 'DFF']
        self.dff_d = {w: gates[w]['inputs'][0] for w in self.dffs}
        dff_set = set(self.dffs)

        # primary inputs split into "real" inputs and key inputs
        self.outputs = list(outputs)
        self.key_inputs = sorted(
            (w for w in inputs if w.lower().startswith('keyinput')),
            key=_key_index,
        )
        self.inputs = [w for w in inputs if not w.lower().startswith('keyinput')]

        self.is_sequential = bool(self.dffs)
        self.order = self._topo(dff_set)

    def _topo(self, dff_set):
        sources = set(self.inputs) | set(self.key_inputs) | dff_set | _CONST
        comb = [w for w in self.gates if w not in dff_set]
        indeg = {w: 0 for w in comb}
        fanout = defaultdict(list)
        for w in comb:
            for inp in self.gates[w]['inputs']:
                if inp in self.gates and inp not in dff_set:
                    indeg[w] += 1
                    fanout[inp].append(w)
        q = deque(w for w in comb if indeg[w] == 0)
        order, seen = [], set()
        while q:
            w = q.popleft()
            if w in seen:
                continue
            seen.add(w)
            order.append(w)
            for nxt in fanout[w]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        # combinational feedback among non-DFF gates would be malformed; append anyway
        order.extend(w for w in comb if w not in seen)
        return order


def _key_index(name):
    digits = ''.join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def parse_bench(path):
    inputs, outputs, gates = [], [], {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('INPUT('):
                inputs.append(line[line.index('(') + 1:line.rindex(')')].strip())
            elif up.startswith('OUTPUT('):
                outputs.append(line[line.index('(') + 1:line.rindex(')')].strip())
            elif '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                gt = rhs[:rhs.index('(')].strip().upper()
                gt = _ALIAS.get(gt, gt)
                args = rhs[rhs.index('(') + 1:rhs.rindex(')')]
                gates[lhs] = {'type': gt,
                              'inputs': [a.strip() for a in args.split(',') if a.strip()]}
    return Circuit(inputs, outputs, gates)


# ---------------------------------------------------------------------------
# Plain simulation (the oracle, and final verification)
# ---------------------------------------------------------------------------

def _eval_gate(gt, vals):
    if gt == 'AND':
        return int(all(vals))
    if gt == 'NAND':
        return 1 - int(all(vals))
    if gt == 'OR':
        return int(any(vals))
    if gt == 'NOR':
        return 1 - int(any(vals))
    if gt in ('NOT',):
        return 1 - vals[0]
    if gt in ('BUF',):
        return vals[0]
    if gt == 'XOR':
        r = 0
        for v in vals:
            r ^= v
        return r
    if gt == 'XNOR':
        r = 0
        for v in vals:
            r ^= v
        return 1 - r
    raise ValueError(f"unknown gate type {gt}")


def simulate(circ, input_vals, state=None):
    """One combinational evaluation. Returns (outputs, next_state)."""
    val = {'0': 0, '1': 1, "1'b0": 0, "1'b1": 1}
    val.update(input_vals)
    if state:
        val.update(state)
    else:
        for d in circ.dffs:
            val.setdefault(d, 0)
    for w in circ.order:
        g = circ.gates[w]
        val[w] = _eval_gate(g['type'], [val[i] for i in g['inputs']])
    outs = {o: val[o] for o in circ.outputs}
    nxt = {d: val[circ.dff_d[d]] for d in circ.dffs}
    return outs, nxt


def oracle_response(oracle, input_seq):
    """Run the oracle from reset over a sequence of input dicts.

    Returns a list of output dicts, one per cycle.
    """
    state = {d: 0 for d in oracle.dffs}
    out_seq = []
    for iv in input_seq:
        outs, state = simulate(oracle, iv, state)
        out_seq.append(outs)
    return out_seq


# ---------------------------------------------------------------------------
# CNF encoder with constant propagation  (the "crunch")
# ---------------------------------------------------------------------------

# A wire value is one of:
#   0           constant false
#   1           constant true
#   ('L', lit)  a SAT literal (possibly negative)
F = 0
T = 1


def _is_const(v):
    return v is F or v is T


class Enc:
    """Tseitin encoder that constant-folds while building, feeding clauses
    straight into a persistent incremental Glucose4 solver."""

    def __init__(self):
        self.pool = IDPool()
        self.solver = Glucose4(incr=True)
        self.n_clauses = 0

    def newvar(self):
        return self.pool.id()

    def add(self, clause):
        self.solver.add_clause(clause)
        self.n_clauses += 1

    @staticmethod
    def lit(v):
        return v[1]

    @staticmethod
    def NEG(v):
        if v is F:
            return T
        if v is T:
            return F
        return ('L', -v[1])

    def AND(self, vals):
        keep = []
        for v in vals:
            if v is F:
                return F
            if v is T:
                continue
            keep.append(v)
        if not keep:
            return T
        if len(keep) == 1:
            return keep[0]
        y = self.newvar()
        lits = [self.lit(v) for v in keep]
        for li in lits:
            self.add([-y, li])
        self.add([y] + [-li for li in lits])
        return ('L', y)

    def OR(self, vals):
        keep = []
        for v in vals:
            if v is T:
                return T
            if v is F:
                continue
            keep.append(v)
        if not keep:
            return F
        if len(keep) == 1:
            return keep[0]
        y = self.newvar()
        lits = [self.lit(v) for v in keep]
        for li in lits:
            self.add([y, -li])
        self.add([-y] + lits)
        return ('L', y)

    def _xor2(self, a, b):
        la, lb = self.lit(a), self.lit(b)
        y = self.newvar()
        self.add([-y, -la, -lb])
        self.add([-y, la, lb])
        self.add([y, -la, lb])
        self.add([y, la, -lb])
        return ('L', y)

    def XOR(self, vals):
        parity = 0
        syms = []
        for v in vals:
            if v is F:
                continue
            if v is T:
                parity ^= 1
                continue
            syms.append(v)
        if not syms:
            return T if parity else F
        acc = syms[0]
        for v in syms[1:]:
            acc = self._xor2(acc, v)
        return self.NEG(acc) if parity else acc

    def gate(self, gt, vals):
        if gt == 'AND':
            return self.AND(vals)
        if gt == 'NAND':
            return self.NEG(self.AND(vals))
        if gt == 'OR':
            return self.OR(vals)
        if gt == 'NOR':
            return self.NEG(self.OR(vals))
        if gt == 'NOT':
            return self.NEG(vals[0])
        if gt == 'BUF':
            return vals[0]
        if gt == 'XOR':
            return self.XOR(vals)
        if gt == 'XNOR':
            return self.NEG(self.XOR(vals))
        raise ValueError(f"unknown gate type {gt}")

    def fix(self, v, const):
        """Force value v to a 0/1 constant via a unit clause."""
        if _is_const(v):
            if v != const:
                self.add([])               # unsatisfiable: contradiction
            return
        lit = self.lit(v)
        self.add([lit] if const else [-lit])

    # --- model helpers -----------------------------------------------------
    def value_of(self, v, model_set):
        if _is_const(v):
            return v
        return 1 if self.lit(v) in model_set else 0


def encode_frame(enc, circ, keymap, input_vals, state_vals):
    """Encode one combinational image of `circ`.

    keymap/input_vals/state_vals map wire-name -> value (const or ('L',lit)).
    Returns (outputs, next_state) as name -> value dicts.
    """
    val = {'0': F, '1': T, "1'b0": F, "1'b1": T}
    val.update(input_vals)
    val.update(keymap)
    val.update(state_vals)
    for w in circ.order:
        g = circ.gates[w]
        val[w] = enc.gate(g['type'], [val[i] for i in g['inputs']])
    outs = {o: val[o] for o in circ.outputs}
    nxt = {d: val[circ.dff_d[d]] for d in circ.dffs}
    return outs, nxt


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------

class KC2Attack:
    def __init__(self, locked, oracle, crunch=True, verbose=True):
        if locked.inputs != oracle.inputs:
            # tolerate ordering, but the PI *sets* must match for oracle queries
            if set(locked.inputs) != set(oracle.inputs):
                raise ValueError(
                    "locked and oracle primary-input sets differ; cannot query oracle")
        self.locked = locked
        self.oracle = oracle
        self.crunch = crunch
        self.verbose = verbose
        self.enc = Enc()
        self.queries = 0

        # one shared set of key variables per mitter side
        self.k1 = {k: ('L', self.enc.newvar()) for k in locked.key_inputs}
        self.k2 = {k: ('L', self.enc.newvar()) for k in locked.key_inputs}

    def log(self, *a):
        if self.verbose:
            print(*a)
            sys.stdout.flush()

    # -- shared bits --------------------------------------------------------
    def _fresh_inputs(self):
        return {pi: ('L', self.enc.newvar()) for pi in self.locked.inputs}

    def _add_io_constraint(self, input_seq, out_seq):
        """Conjoin  ce(I,k1)=O  and  ce(I,k2)=O  for an observed I/O (sequence).

        Inputs/outputs are constants -> the circuit copy is constant-folded and
        only the key-dependent logic survives in CNF (key-condition crunching).
        """
        for key in (self.k1, self.k2):
            state = {d: F for d in self.locked.dffs}        # reset
            for iv, ov in zip(input_seq, out_seq):
                ivals = {pi: (T if iv[pi] else F) for pi in self.locked.inputs}
                outs, state = encode_frame(self.enc, self.locked, key, ivals, state)
                for o in self.locked.outputs:
                    self.enc.fix(outs[o], ov[o])

    def _generalize_block(self, key_assign):
        """Negative-key-condition compression (sec. III-E).

        Add a blocking clause forbidding the disqualified candidate key, then
        try to drop literals (generalize the cube) while the solver still
        proves the reduced clause is implied by the accumulated conditions.
        """
        lits = []
        for k in self.locked.key_inputs:
            v = self.k1[k]
            lit = self.enc.lit(v)
            lits.append(-lit if key_assign[k] else lit)   # clause = NOT(this key)
        # cheap generalization: drop a literal if the rest is already entailed
        i = 0
        while i < len(lits) and len(lits) > 1:
            trial = lits[:i] + lits[i + 1:]
            # entailed if asserting the negation of `trial` together with F is UNSAT
            assumptions = [-l for l in trial]
            if not self.enc.solver.solve(assumptions=assumptions):
                lits = trial
            else:
                i += 1
        self.enc.add(lits)

    # -- combinational attack ----------------------------------------------
    def run_combinational(self, max_iter):
        enc = self.enc
        ivars = self._fresh_inputs()
        o1, _ = encode_frame(enc, self.locked, self.k1, ivars, {})
        o2, _ = encode_frame(enc, self.locked, self.k2, ivars, {})
        diffs = [enc.XOR([o1[o], o2[o]]) for o in self.locked.outputs]
        mitter = enc.OR(diffs)
        if mitter is F:
            self.log("[kc2] keys provably irrelevant to output; any key works.")
            return self._extract_key()
        sel = enc.lit(mitter)                              # assume the mitter holds

        while self.queries < max_iter:
            if not enc.solver.solve(assumptions=[sel]):
                break                                      # no more DIPs -> done
            model = set(enc.solver.get_model())
            dip = {pi: enc.value_of(ivars[pi], model) for pi in self.locked.inputs}
            cand1 = {k: enc.value_of(self.k1[k], model) for k in self.locked.key_inputs}
            cand2 = {k: enc.value_of(self.k2[k], model) for k in self.locked.key_inputs}

            ov = oracle_response(self.oracle, [dip])[0]
            self._add_io_constraint([dip], [ov])
            self.queries += 1

            if self.crunch:
                # whichever candidate disagrees with the oracle is now dead
                lo, _ = simulate(self.locked, {**dip, **_keymap_int(cand1)})
                if any(lo[o] != ov[o] for o in self.locked.outputs):
                    self._generalize_block(cand1)
                lo, _ = simulate(self.locked, {**dip, **_keymap_int(cand2)})
                if any(lo[o] != ov[o] for o in self.locked.outputs):
                    self._generalize_block(cand2)

            if self.queries % 25 == 0:
                self.log(f"[kc2] {self.queries} DIPs, {enc.n_clauses} clauses")

        return self._extract_key()

    # -- sequential attack (Algorithm 1 + KC2) ------------------------------
    def run_sequential(self, max_iter, max_depth):
        enc = self.enc
        self.frame_inputs = []          # symbolic inputs, one dict per frame
        self.frame_diffs = []           # per-frame list of output-XOR values
        self._st1 = {d: F for d in self.locked.dffs}     # reset (R)
        self._st2 = {d: F for d in self.locked.dffs}
        self._built = 0
        self._ce = self._build_ce_gadget()

        b = 1
        self._extend_unrolling(b)
        while self.queries < max_iter:
            sel = self._mitter_property(b)
            if enc.solver.solve(assumptions=[sel]):
                # a discriminating input sequence exists within bound b
                model = set(enc.solver.get_model())
                dis = [{pi: enc.value_of(self.frame_inputs[t][pi], model)
                        for pi in self.locked.inputs} for t in range(b)]
                out_seq = oracle_response(self.oracle, dis)
                self._add_io_constraint(dis, out_seq)
                self.queries += 1
                if self.queries % 10 == 0:
                    self.log(f"[kc2] {self.queries} DISes, depth {b}, "
                             f"{enc.n_clauses} clauses")
            else:
                # no disagreement up to depth b: check CE termination
                if not enc.solver.solve(assumptions=[self._ce]):
                    self.log(f"[kc2] combinational-equivalence termination at depth {b}")
                    break
                if b >= max_depth:
                    self.log(f"[kc2] reached max unroll depth {max_depth}; stopping")
                    break
                b *= 2
                self._extend_unrolling(b)
                self.log(f"[kc2] extending unroll depth -> {b}")
        return self._extract_key()

    def _extend_unrolling(self, b):
        enc = self.enc
        while self._built < b:
            ivars = self._fresh_inputs()
            self.frame_inputs.append(ivars)
            o1, self._st1 = encode_frame(enc, self.locked, self.k1, ivars, self._st1)
            o2, self._st2 = encode_frame(enc, self.locked, self.k2, ivars, self._st2)
            self.frame_diffs.append([enc.XOR([o1[o], o2[o]])
                                     for o in self.locked.outputs])
            self._built += 1

    def _mitter_property(self, b):
        """Selector literal s such that assuming s asserts 'some output of some
        frame in 0..b-1 disagrees' (the BMC safety violation G(not M))."""
        enc = self.enc
        s = enc.newvar()
        all_diffs = [d for t in range(b) for d in self.frame_diffs[t]]
        clause = [-s] + [enc.lit(d) for d in all_diffs if not _is_const(d)]
        if any(d is T for d in all_diffs):
            return T_SELECT(enc)        # trivially satisfiable; should not happen
        enc.add(clause)
        return s

    def _build_ce_gadget(self):
        """A single transition image from an *unconstrained* start state that
        asserts k1 and k2 differ in output or next-state. UNSAT (under the
        accumulated key conditions) => combinational equivalence => terminate."""
        enc = self.enc
        ivars = self._fresh_inputs()
        st1 = {d: ('L', enc.newvar()) for d in self.locked.dffs}
        st2 = {d: st1[d] for d in self.locked.dffs}       # same start state
        o1, n1 = encode_frame(enc, self.locked, self.k1, ivars, st1)
        o2, n2 = encode_frame(enc, self.locked, self.k2, ivars, st2)
        diffs = [enc.XOR([o1[o], o2[o]]) for o in self.locked.outputs]
        diffs += [enc.XOR([n1[d], n2[d]]) for d in self.locked.dffs]
        s = enc.newvar()
        lits = [enc.lit(d) for d in diffs if not _is_const(d)]
        enc.add([-s] + lits)
        return s

    # -- finish -------------------------------------------------------------
    def _extract_key(self):
        if not self.enc.solver.solve():
            raise RuntimeError("accumulated key conditions are UNSAT (bug or bad oracle)")
        model = set(self.enc.solver.get_model())
        key_bits = {k: self.enc.value_of(self.k1[k], model)
                    for k in self.locked.key_inputs}
        return key_bits


def _keymap_int(assign):
    return {k: int(v) for k, v in assign.items()}


def T_SELECT(enc):
    s = enc.newvar()
    enc.add([s])
    return s


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(locked, oracle, key_bits, n_patterns=400, seq_len=40, seed=0):
    rng = random.Random(seed)
    km = _keymap_int(key_bits)
    if locked.is_sequential:
        for _ in range(n_patterns // seq_len + 1):
            seq = [{pi: rng.randint(0, 1) for pi in oracle.inputs}
                   for _ in range(seq_len)]
            exp = oracle_response(oracle, seq)
            ls, gs = {d: 0 for d in locked.dffs}, None
            for t, iv in enumerate(seq):
                louts, ls = simulate(locked, {**iv, **km}, ls)
                if any(louts[o] != exp[t][o] for o in locked.outputs):
                    return False
        return True
    for _ in range(n_patterns):
        iv = {pi: rng.randint(0, 1) for pi in oracle.inputs}
        exp, _ = simulate(oracle, iv)
        got, _ = simulate(locked, {**iv, **km})
        if any(got[o] != exp[o] for o in locked.outputs):
            return False
    return True


def key_to_int(circ, key_bits):
    val = 0
    for k in circ.key_inputs:
        val |= (key_bits[k] & 1) << _key_index(k)
    return val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KC2 oracle-guided deobfuscation attack")
    ap.add_argument('locked', help='locked .bench file')
    ap.add_argument('--oracle', required=True, help='original (unlocked) .bench file')
    ap.add_argument('--max-iter', type=int, default=100000, help='max DIP/DIS queries')
    ap.add_argument('--max-depth', type=int, default=512, help='max sequential unroll depth')
    ap.add_argument('--no-crunch', action='store_true', help='disable key-condition crunching')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for verification')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    locked = parse_bench(args.locked)
    oracle = parse_bench(args.oracle)

    mode = 'sequential' if locked.is_sequential else 'combinational'
    if not args.quiet:
        print(f"[kc2] locked  : {args.locked}")
        print(f"[kc2] oracle  : {args.oracle}")
        print(f"[kc2] mode    : {mode}  "
              f"({len(locked.inputs)} inputs, {len(locked.outputs)} outputs, "
              f"{len(locked.dffs)} dffs, {len(locked.key_inputs)} key bits)")

    attack = KC2Attack(locked, oracle, crunch=not args.no_crunch, verbose=not args.quiet)
    t0 = time.time()
    if locked.is_sequential:
        key_bits = attack.run_sequential(args.max_iter, args.max_depth)
    else:
        key_bits = attack.run_combinational(args.max_iter)
    elapsed = time.time() - t0

    ok = verify(locked, oracle, key_bits, seed=args.seed)
    kint = key_to_int(locked, key_bits)
    width = (len(locked.key_inputs) + 3) // 4

    print("=" * 60)
    print("  KC2 attack — result")
    print("=" * 60)
    print(f"  queries (DIP/DIS) : {attack.queries}")
    print(f"  clauses learned   : {attack.enc.n_clauses}")
    print(f"  runtime           : {elapsed:.3f} s")
    bits = ''.join(str(key_bits[k]) for k in reversed(locked.key_inputs))
    print(f"  recovered key bin : {bits}")
    print(f"  recovered key hex : 0x{kint:0{max(width,1)}X}")
    print(f"  verified vs oracle: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
