"""
python epic_lock.py <input.bench> [--key-bits N] [--key HEX] [--output out.bench]
"""

import argparse
import random
import sys
from collections import defaultdict, deque


"""
goes line by line and sorts each line into one of 4 data structures based on the syntax of the line
returns a dict with keys:
    inputs defined by the keyword 'INPUT' : list of str
    outputs defined by the keyword 'OUTPUT: list of str
    gates defined by the RHS of = : dict  wire_name -> {'type': str, 'inputs': list[str]}
    comments : list of str
"""

def parse_bench(path: str) -> dict:
    inputs, outputs, gates, comments = [], [], {}, []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                comments.append(line)
                continue

            up = line.upper()
            if up.startswith('INPUT('):
                name = line[line.index('(') + 1: line.index(')')].strip()
                inputs.append(name)
            elif up.startswith('OUTPUT('):
                name = line[line.index('(') + 1: line.index(')')].strip()
                outputs.append(name)
            else:
                if '=' not in line:
                    continue
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                paren_open = rhs.index('(')
                gate_type = rhs[:paren_open].strip().upper()
                args_str = rhs[paren_open + 1: rhs.rindex(')')].strip()
                gate_inputs = [a.strip() for a in args_str.split(',') if a.strip()]
                gates[lhs] = {'type': gate_type, 'inputs': gate_inputs}

    return {'inputs': inputs, 'outputs': outputs,
            'gates': gates, 'comments': comments}


"""
graph helper function needed to walk the graph forward in topological sorting. the gate dict is inverted so that rather than 
the gate output -> input, it becomes wire -> gates that it uses as an input
"""
def build_fanout(gates: dict) -> dict:
    fanout = defaultdict(list)
    for out, g in gates.items():
        for inp in g['inputs']:
            fanout[inp].append(out)
    return fanout


"""
Implements Kahn's BFS algorithm. Starting with each gate that have all primary inputs, it repeated pops the gates from the queue
and removes the wire name. The in-degree of every gate that fans into the specific gate is reduced. When the in-degree of a gate reaches
0, it's added to the queue.
"""
def topological_order(gates: dict, primary_inputs: list) -> list:
    in_degree = {w: 0 for w in gates}
    fanout = defaultdict(list)
    for out, g in gates.items():
        for inp in g['inputs']:
            if inp in gates:          # only count gate-to-gate edges
                in_degree[out] += 1
                fanout[inp].append(out)
    queue = deque()
    # seeds: primary inputs resolve to fan-in = 0 for their driven gates
    # actually seed with gates whose every input is a primary input or const
    pi_set = set(primary_inputs) | {'1', '0', "1'b0", "1'b1"}
    for w, g in gates.items():
        if all(inp in pi_set or inp not in gates for inp in g['inputs']):
            queue.append(w)
    order = []
    visited = set()
    while queue:
        w = queue.popleft()
        if w in visited:
            continue
        visited.add(w)
        order.append(w)
        for nxt in fanout[w]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    # catch any remaining cycles
    for w in gates:
        if w not in visited:
            order.append(w)
    return order


"""
walking down the topological order, each wire is assigned a numeric depth with inputs being 0. each gate's depth is one more than
the max depth of its inputs, determining how deep a wire is situated in the critical path. wires near outputs have the highest level
"""
def estimate_levels(gates: dict, primary_inputs: list) -> dict:
    level = {pi: 0 for pi in primary_inputs}
    level.update({'0': 0, '1': 0, "1'b0": 0, "1'b1": 0})

    for w in topological_order(gates, primary_inputs):
        g = gates[w]
        max_inp = max((level.get(inp, 0) for inp in g['inputs']), default=0)
        level[w] = max_inp + 1
    return level

"""
filters out the directly observable primary outputs and any wires that are in the top 10% of the logic depth.
the remaining candidates are sorted by descending depth as deeper wires propagate greater. N-wires equal to the
number of key bits are randomly sampled from the top half of the sorted list
"""
def select_wires(gates: dict, primary_inputs: list, primary_outputs: list,
                 key_bits: int, seed: int | None = None) -> list:
    rng = random.Random(seed)
    pi_set = set(primary_inputs)
    po_set = set(primary_outputs)

    level = estimate_levels(gates, primary_inputs)
    max_level = max(level.get(w, 0) for w in gates) if gates else 1

    # exclude POs and high level wires
    threshold = max_level * 0.90
    candidates = [
        w for w in gates
        if w not in po_set
        and level.get(w, 0) < threshold
        and gates[w]['type'] not in ('DFF',)   # skip flip flop outputs
    ]

    if len(candidates) < key_bits:
        # allow all non-PI wires
        candidates = [w for w in gates if w not in pi_set]

    if len(candidates) < key_bits:
        raise ValueError(
            f"Circuit has only {len(candidates)} candidate wires "
            f"but {key_bits} key bits requested."
        )
    candidates.sort(key=lambda w: -level.get(w, 0))
    top = candidates[:max(key_bits * 4, len(candidates) // 2)]
    return rng.sample(top, key_bits)


"""
for each selected wire and its corresponding key bit, the wire's gate definition is renamed to preserve the 
original computation of a wire without changing its name downstream. A new gate is defined for each wire such that any other bit 
other than the correct key bit essentially inverts the wire

Returns:
    locked_parsed : modified circuit dict
    key_value     : integer key (correct unlock value)
    locked_wires  : list of (wire, key_bit_index, gate_type)
"""
def lock_circuit(parsed: dict, key_bits: int,
                 key: int | None = None,
                 seed: int | None = None) -> tuple[dict, int, list]:
    gates = {k: dict(v, inputs=list(v['inputs']))
             for k, v in parsed['gates'].items()}
    inputs = list(parsed['inputs'])
    outputs = list(parsed['outputs'])

    rng = random.Random(seed)

    # generate random key if not provided
    if key is None:
        key = rng.getrandbits(key_bits)

    selected = select_wires(gates, inputs, outputs, key_bits, seed)

    locked_wires = []
    new_key_inputs = []

    for idx, wire in enumerate(selected):
        key_bit = (key >> idx) & 1
        key_pin = f'keyinput{idx}'
        new_key_inputs.append(key_pin)

        original_gate = gates[wire]

        # insert a "shadow" wire that carries the original gate's output
        shadow = f'{wire}_EPIC_orig'
        gates[shadow] = original_gate   
        gate_type = 'XOR' if key_bit == 0 else 'XNOR'
        gates[wire] = {'type': gate_type, 'inputs': [shadow, key_pin]}

        locked_wires.append((wire, idx, gate_type, key_bit))

    # rewrite any reference to a locked wire inside other gate inputs
    inputs = inputs + new_key_inputs

    locked_parsed = {
        'inputs': inputs,
        'outputs': outputs,
        'gates': gates,
        'comments': parsed['comments'],
    }
    return locked_parsed, key, locked_wires

def write_bench(parsed: dict, path: str, header_comments: list | None = None):
    """Write a circuit dict back to .bench format."""
    with open(path, 'w') as fh:
        for c in parsed['comments']:
            fh.write(c + '\n')
        if header_comments:
            for c in header_comments:
                fh.write(c + '\n')
        fh.write('\n')

        for inp in parsed['inputs']:
            fh.write(f'INPUT({inp})\n')
        fh.write('\n')

        for out in parsed['outputs']:
            fh.write(f'OUTPUT({out})\n')
        fh.write('\n')

        for wire, g in parsed['gates'].items():
            args = ', '.join(g['inputs'])
            fh.write(f'{wire} = {g["type"]}({args})\n')

def report(parsed_orig: dict, parsed_locked: dict,
           key_bits: int, key: int, locked_wires: list, output_path: str):
    orig_gates = len(parsed_orig['gates'])
    locked_gates = len(parsed_locked['gates'])
    overhead = locked_gates - orig_gates

    print("=" * 60)
    print("  EPIC Combinational Locking — Summary")
    print("=" * 60)
    print(f"  Original gates   : {orig_gates}")
    print(f"  Locked gates     : {locked_gates}  (+{overhead} XOR/XNOR + shadow)")
    print(f"  Original inputs  : {len(parsed_orig['inputs'])}")
    print(f"  Key inputs added : {key_bits}")
    print(f"  Key length       : {key_bits} bits")
    print(f"  Correct key (hex): 0x{key:0{(key_bits+3)//4}X}")
    print(f"  Correct key (bin): {key:0{key_bits}b}")
    print(f"  Output file      : {output_path}")
    print()
    print("  Locked wires:")
    print(f"  {'Wire':<25} {'KeyBit':>6}  {'Gate':>5}  {'Bit val':>7}")
    print("  " + "-" * 50)
    for wire, idx, gtype, kbit in locked_wires:
        print(f"  {wire:<25} {idx:>6}  {gtype:>5}  {kbit:>7}")
    print("=" * 60)
    print()
    print("  Security note: supply the correct key to all KEY inputs")
    print("  (keyinput0 … keyinput{}) during chip activation.".format(key_bits - 1))
    print("  Without the key, the circuit produces wrong outputs.")
    print("=" * 60)

def main():
    ap = argparse.ArgumentParser(
        description="Apply EPIC combinational locking to an ISCAS .bench file."
    )
    ap.add_argument('input', help='Input .bench file')
    ap.add_argument('--key-bits', type=int, default=32,
                    help='Number of key bits (default: 32; paper recommends ≥64)')
    ap.add_argument('--key', type=lambda x: int(x, 0), default=None,
                    help='Correct key as hex integer, e.g. 0xDEADBEEF '
                         '(random if omitted)')
    ap.add_argument('--output', default=None,
                    help='Output .bench path (default: <input>_locked.bench)')
    ap.add_argument('--seed', type=int, default=None,
                    help='RNG seed for reproducibility')
    args = ap.parse_args()

    if args.output is None:
        base = args.input
        if base.endswith('.bench'):
            base = base[:-6]
        args.output = base + '_locked.bench'

    print(f"[EPIC] Parsing {args.input} …")
    parsed = parse_bench(args.input)
    print(f"       {len(parsed['inputs'])} inputs, "
          f"{len(parsed['outputs'])} outputs, "
          f"{len(parsed['gates'])} gates")

    print(f"[EPIC] Locking with {args.key_bits}-bit key …")
    locked, key, locked_wires = lock_circuit(
        parsed, args.key_bits, key=args.key, seed=args.seed
    )

    write_bench(locked, args.output)
    print(f"[EPIC] Written to {args.output}")
    print()

    report(parsed, locked, args.key_bits, key, locked_wires, args.output)


if __name__ == '__main__':
    main()