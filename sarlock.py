import random
import time
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import itertools

class GateType(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    NAND = "NAND"
    NOR = "NOR"
    XOR = "XOR"
    XNOR = "XNOR"
    BUF = "BUF"


@dataclass
class Gate:
    gate_id: str
    gate_type: GateType
    inputs: List[str]
    output: str
    
    # Computes the output value of this gate given a dict of all current wire values
    def evaluate(self, values: Dict[str, int]) -> int:
        if not all(inp in values for inp in self.inputs):
            raise ValueError(f"Missing input values for gate {self.gate_id}")
        
        input_vals = [values[inp] for inp in self.inputs]
        
        if self.gate_type == GateType.AND:
            return int(all(input_vals))
        elif self.gate_type == GateType.OR:
            return int(any(input_vals))
        elif self.gate_type == GateType.NOT:
            return 1 - input_vals[0]
        elif self.gate_type == GateType.NAND:
            return 1 - int(all(input_vals))
        elif self.gate_type == GateType.NOR:
            return 1 - int(any(input_vals))
        elif self.gate_type == GateType.XOR:
            return int(sum(input_vals) % 2 == 1)
        elif self.gate_type == GateType.XNOR:
            return int(sum(input_vals) % 2 == 0)
        elif self.gate_type == GateType.BUF:
            return input_vals[0]
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")


class Circuit:    
    def __init__(self):
        self.gates: List[Gate] = []
        self.primary_inputs: List[str] = []
        self.primary_outputs: List[str] = []
        self.key_inputs: List[str] = []
        self.wire_to_gate: Dict[str, Gate] = {}
        self.gate_levels: Dict[str, int] = {}
        self._sorted_gates: Optional[List[Gate]] = None  # cached topological order

    def add_gate(self, gate: Gate):
        self.gates.append(gate)
        self.wire_to_gate[gate.output] = gate
        self._sorted_gates = None

    def add_primary_input(self, name: str):
        if name not in self.primary_inputs:
            self.primary_inputs.append(name)
    
    def add_primary_output(self, name: str):
        if name not in self.primary_outputs:
            self.primary_outputs.append(name)
    
    def add_key_input(self, name: str):
        if name not in self.key_inputs:
            self.key_inputs.append(name)
    
    # Assigns each gate a level based on the maximum level of its inputs + 1; primary inputs and key inputs start at level 0. Used to determine evaluation order.
    def compute_levels(self):
        self.gate_levels = {}
        for pi in self.primary_inputs + self.key_inputs:
            self.gate_levels[pi] = 0

        changed = True
        while changed:
            changed = False
            for gate in self.gates:
                max_input_level = -1
                all_inputs_have_level = True
                
                for inp in gate.inputs:
                    if inp in self.gate_levels:
                        max_input_level = max(max_input_level, self.gate_levels[inp])
                    else:
                        all_inputs_have_level = False
                        break
                
                if all_inputs_have_level:
                    new_level = max_input_level + 1
                    if gate.output not in self.gate_levels or self.gate_levels[gate.output] != new_level:
                        self.gate_levels[gate.output] = new_level
                        changed = True

    # Simulates the circuit for given primary input and key values; computes and caches topological gate order on first call, then reuses it
    def evaluate(self, input_values, key_values):
        values = {**input_values, **key_values}
        if self._sorted_gates is None:
            self.compute_levels()
            self._sorted_gates = sorted(self.gates, key=lambda g: self.gate_levels.get(g.output, float('inf')))
        for gate in self._sorted_gates:
            values[gate.output] = gate.evaluate(values)
        return {out: values[out] for out in self.primary_outputs}
    
    # Returns an independent deep copy of the circuit so locking layers can modify it without affecting the original
    def copy(self) -> 'Circuit':
        new_circuit = Circuit()
        new_circuit.primary_inputs = self.primary_inputs.copy()
        new_circuit.primary_outputs = self.primary_outputs.copy()
        new_circuit.key_inputs = self.key_inputs.copy()
        
        for gate in self.gates:
            new_gate = Gate(gate.gate_id, gate.gate_type, gate.inputs.copy(), gate.output)
            new_circuit.add_gate(new_gate)
        
        return new_circuit

class RandomLogicLocking:
    def __init__(self, circuit: Circuit, num_key_bits: int):
        self.original_circuit = circuit
        self.num_key_bits = num_key_bits
        self.locked_circuit: Optional[Circuit] = None
        self.correct_key: Optional[Dict[str, int]] = None
        self.key_gate_locations: List[Tuple[str, str]] = []

    # Inserts XOR/XNOR key gates at randomly chosen internal wires; correct key bit = 0 means XOR (transparent), 1 means XNOR (transparent)
    def lock(self, seed: Optional[int] = None) -> Circuit:
        if seed is not None:
            random.seed(seed)
        
        self.locked_circuit = self.original_circuit.copy()
        self.correct_key = {f"keyinput{i}": random.randint(0, 1) for i in range(self.num_key_bits)}
        
        for key_name in self.correct_key.keys():
            self.locked_circuit.add_key_input(key_name)
        
        internal_wires = []
        for gate in self.locked_circuit.gates:
            if gate.output not in self.locked_circuit.primary_outputs:
                internal_wires.append(gate.output)
        
        if len(internal_wires) < self.num_key_bits:
            raise ValueError(f"Not enough internal wires ({len(internal_wires)}) for {self.num_key_bits} key bits")
        
        selected_wires = random.sample(internal_wires, self.num_key_bits)
        
        for i, wire in enumerate(selected_wires):
            key_name = f"keyinput{i}"
            use_xor = random.choice([True, False])
            gate_type = GateType.XOR if use_xor else GateType.XNOR
            new_wire = f"{wire}_locked"
            
            for gate in self.locked_circuit.gates:
                gate.inputs = [new_wire if inp == wire else inp for inp in gate.inputs]
            
            self.locked_circuit.primary_outputs = [new_wire if out == wire else out for out in self.locked_circuit.primary_outputs]
            
            key_gate = Gate(
                gate_id=f"KEY_GATE_{i}",
                gate_type=gate_type,
                inputs=[wire, key_name],
                output=new_wire
            )
            self.locked_circuit.add_gate(key_gate)
            self.key_gate_locations.append((wire, key_name))
        
        return self.locked_circuit


class StrongLogicLocking:
    def __init__(self, circuit: Circuit, num_key_bits: int):
        self.original_circuit = circuit
        self.num_key_bits = num_key_bits
        self.locked_circuit: Optional[Circuit] = None
        self.correct_key: Optional[Dict[str, int]] = None

    # Inserts key gates distributed across different circuit levels to maximize interference; alternates XOR/XNOR to make individual key bits harder to sensitize
    def lock(self, seed: Optional[int] = None) -> Circuit:
        if seed is not None:
            random.seed(seed)
        
        self.locked_circuit = self.original_circuit.copy()
        self.correct_key = {f"keyinput{i}": random.randint(0, 1) for i in range(self.num_key_bits)}
        
        for key_name in self.correct_key.keys():
            self.locked_circuit.add_key_input(key_name)
        
        self.locked_circuit.compute_levels()
        
        # Group gates by level to distribute key gates across the circuit depth
        gates_by_level: Dict[int, List[Gate]] = {}
        for gate in self.locked_circuit.gates:
            level = self.locked_circuit.gate_levels.get(gate.output, 0)
            if level not in gates_by_level:
                gates_by_level[level] = []
            gates_by_level[level].append(gate)
        
        selected_wires = []
        levels = sorted(gates_by_level.keys())
        
        for i in range(self.num_key_bits):
            level_idx = i % len(levels)
            level = levels[level_idx]
            candidates = [g.output for g in gates_by_level[level] 
                         if g.output not in self.locked_circuit.primary_outputs]
            
            if candidates:
                wire = random.choice(candidates)
                selected_wires.append(wire)
            else:
                internal_wires = [g.output for g in self.locked_circuit.gates 
                                 if g.output not in self.locked_circuit.primary_outputs]
                if internal_wires:
                    selected_wires.append(random.choice(internal_wires))
        
        for i, wire in enumerate(selected_wires[:self.num_key_bits]):
            key_name = f"keyinput{i}"
            gate_type = GateType.XOR if i % 2 == 0 else GateType.XNOR
            new_wire = f"{wire}_sll_{i}"
            
            for gate in self.locked_circuit.gates:
                gate.inputs = [new_wire if inp == wire else inp for inp in gate.inputs]
            
            self.locked_circuit.primary_outputs = [new_wire if out == wire else out 
                                                   for out in self.locked_circuit.primary_outputs]
            
            key_gate = Gate(
                gate_id=f"SLL_GATE_{i}",
                gate_type=gate_type,
                inputs=[wire, key_name],
                output=new_wire
            )
            self.locked_circuit.add_gate(key_gate)
        
        return self.locked_circuit


class SARLock:
    def __init__(self, circuit: Circuit, num_key_bits: int):
        self.original_circuit = circuit
        self.num_key_bits = num_key_bits
        self.locked_circuit: Optional[Circuit] = None
        self.correct_key: Optional[Dict[str, int]] = None
        self.flip_output: Optional[str] = None

    # Builds a comparator that outputs 1 when all input bits match their corresponding key bits; implemented as XNOR gates feeding an AND tree
    def _build_comparator(self, input_wires: List[str], key_wires: List[str], 
                         gate_prefix: str) -> Tuple[str, List[Gate]]:
        gates = []
        xnor_outputs = []
        for i, (inp, key) in enumerate(zip(input_wires, key_wires)):
            xnor_wire = f"{gate_prefix}_xnor_{i}"
            xnor_gate = Gate(
                gate_id=f"{gate_prefix}_XNOR_{i}",
                gate_type=GateType.XNOR,
                inputs=[inp, key],
                output=xnor_wire
            )
            gates.append(xnor_gate)
            xnor_outputs.append(xnor_wire)
        
        # AND tree reduces all XNOR outputs to a single match signal
        current_wires = xnor_outputs
        counter = 0
        while len(current_wires) > 1:
            next_wires = []
            for i in range(0, len(current_wires), 2):
                if i + 1 < len(current_wires):
                    and_wire = f"{gate_prefix}_and_{counter}"
                    and_gate = Gate(
                        gate_id=f"{gate_prefix}_AND_{counter}",
                        gate_type=GateType.AND,
                        inputs=[current_wires[i], current_wires[i+1]],
                        output=and_wire
                    )
                    gates.append(and_gate)
                    next_wires.append(and_wire)
                    counter += 1
                else:
                    next_wires.append(current_wires[i])
            current_wires = next_wires
        
        return current_wires[0], gates

    # Inverts the comparator output so the flip signal is 1 on mismatch and 0 on match (suppressing the flip when the correct key is applied)
    def _build_mask(self, comparator_out: str, gate_prefix: str) -> Tuple[str, List[Gate]]:
        gates = []
        not_wire = f"{gate_prefix}_mask_out"
        not_gate = Gate(
            gate_id=f"{gate_prefix}_MASK_NOT",
            gate_type=GateType.NOT,
            inputs=[comparator_out],
            output=not_wire
        )
        gates.append(not_gate)
        return not_wire, gates

    # Applies SARLock: builds comparator + mask and XORs the flip signal into the target output; forces 2^n - 1 DIPs for a successful SAT attack
    def lock(self, target_output_idx: int = 0, seed: Optional[int] = None,
             use_standard_key_names: bool = True) -> Circuit:
        if seed is not None:
            random.seed(seed)
        
        self.locked_circuit = self.original_circuit.copy()
        
        if use_standard_key_names:
            key_names = [f"keyinput{i}" for i in range(self.num_key_bits)]
        else:
            key_names = [f"SK{i}" for i in range(self.num_key_bits)]
        
        self.correct_key = {key_names[i]: random.randint(0, 1) for i in range(self.num_key_bits)}
        
        for key_name in key_names:
            self.locked_circuit.add_key_input(key_name)
        
        # Compare only as many primary inputs as there are key bits
        num_compare_bits = min(self.num_key_bits, len(self.locked_circuit.primary_inputs))
        compare_inputs = self.locked_circuit.primary_inputs[:num_compare_bits]
        compare_keys = key_names[:num_compare_bits]
        
        comp_out, comp_gates = self._build_comparator(compare_inputs, compare_keys, "SARLOCK")
        for gate in comp_gates:
            self.locked_circuit.add_gate(gate)
        
        mask_out, mask_gates = self._build_mask(comp_out, "SARLOCK")
        for gate in mask_gates:
            self.locked_circuit.add_gate(gate)
        
        # XOR the flip signal into the chosen primary output
        target_output = self.locked_circuit.primary_outputs[target_output_idx]
        new_output = f"{target_output}_sarlock"
        xor_gate = Gate(
            gate_id="SARLOCK_OUTPUT_XOR",
            gate_type=GateType.XOR,
            inputs=[target_output, mask_out],
            output=new_output
        )
        self.locked_circuit.add_gate(xor_gate)
        self.locked_circuit.primary_outputs[target_output_idx] = new_output
        self.flip_output = new_output
        
        return self.locked_circuit


class SARLock_SLL:
    def __init__(self, circuit: Circuit, num_key_bits_sll: int, num_key_bits_sarlock: int):
        self.original_circuit = circuit
        self.num_key_bits_sll = num_key_bits_sll
        self.num_key_bits_sarlock = num_key_bits_sarlock
        self.locked_circuit: Optional[Circuit] = None
        self.correct_key: Optional[Dict[str, int]] = None
        self.sll: Optional[StrongLogicLocking] = None
        self.sarlock: Optional[SARLock] = None

    # Applies SLL first then SARLock on top; renames all key inputs to a unified keyinput0..N sequence so the output is compatible with SAT tools;
    # K1 (SLL) defends against sensitization/removal, K2 (SARLock) defeats SAT attack
    def lock(self, seed: Optional[int] = None) -> Circuit:
        if seed is not None:
            random.seed(seed)
        
        # Layer 1: SLL
        self.sll = StrongLogicLocking(self.original_circuit, self.num_key_bits_sll)
        sll_locked = self.sll.lock(seed)
        
        # Rename SLL keys to keyinput0..N-1
        old_keys = sorted(self.sll.correct_key.keys())
        self.correct_key = {}
        for i, old_key in enumerate(old_keys):
            new_key = f"keyinput{i}"
            self.correct_key[new_key] = self.sll.correct_key[old_key]
            sll_locked.key_inputs = [new_key if k == old_key else k for k in sll_locked.key_inputs]
            for gate in sll_locked.gates:
                gate.inputs = [new_key if inp == old_key else inp for inp in gate.inputs]
        
        # Layer 2: SARLock using temporary SK* key names to avoid collisions
        self.sarlock = SARLock(sll_locked, self.num_key_bits_sarlock)
        sarlock_circuit = self.sarlock.lock(seed=seed, use_standard_key_names=False)
        
        # Rename SARLock keys to continue numbering from where SLL left off
        offset = self.num_key_bits_sll
        old_sarlock_keys = sorted([k for k in sarlock_circuit.key_inputs if k.startswith('SK')])
        for i, old_key in enumerate(old_sarlock_keys):
            new_key = f"keyinput{offset + i}"
            self.correct_key[new_key] = self.sarlock.correct_key[old_key]
            sarlock_circuit.key_inputs = [new_key if k == old_key else k for k in sarlock_circuit.key_inputs]
            for gate in sarlock_circuit.gates:
                gate.inputs = [new_key if inp == old_key else inp for inp in gate.inputs]
        
        self.locked_circuit = sarlock_circuit
        return self.locked_circuit


class BenchmarkParser:

    # Reads an ISCAS .bench file and constructs a Circuit with all inputs,
    # outputs, and gates populated from the file
    @staticmethod
    def parse_bench(filename: str) -> Circuit:
        circuit = Circuit()
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('INPUT'):
                input_name = line.split('(')[1].split(')')[0]
                circuit.add_primary_input(input_name)
            elif line.startswith('OUTPUT'):
                output_name = line.split('(')[1].split(')')[0]
                circuit.add_primary_output(output_name)
            elif '=' in line:
                parts = line.split('=')
                output = parts[0].strip()
                gate_expr = parts[1].strip()
                
                if '(' in gate_expr:
                    gate_type_str = gate_expr.split('(')[0].strip()
                    inputs_str = gate_expr.split('(')[1].split(')')[0]
                    inputs = [inp.strip() for inp in inputs_str.split(',')]
                    
                    gate_type_map = {
                        'AND': GateType.AND, 'OR': GateType.OR, 'NOT': GateType.NOT,
                        'NAND': GateType.NAND, 'NOR': GateType.NOR, 'XOR': GateType.XOR,
                        'XNOR': GateType.XNOR, 'BUF': GateType.BUF, 'BUFF': GateType.BUF,
                    }
                    
                    if gate_type_str in gate_type_map:
                        gate = Gate(
                            gate_id=f"G_{output}",
                            gate_type=gate_type_map[gate_type_str],
                            inputs=inputs,
                            output=output
                        )
                        circuit.add_gate(gate)
        
        return circuit

    # Serializes a Circuit back to .bench format with all inputs (primary + key),outputs, and gates written in standard ISCAS style
    @staticmethod
    def write_bench(circuit: Circuit, filename: str):
        gate_type_str_map = {
            GateType.AND: 'AND', GateType.OR: 'OR', GateType.NOT: 'NOT',
            GateType.NAND: 'NAND', GateType.NOR: 'NOR', GateType.XOR: 'XOR',
            GateType.XNOR: 'XNOR', GateType.BUF: 'BUF', }
        
        with open(filename, 'w') as f:
            f.write("# Locked circuit\n")
            f.write("# Generated by SARLock implementation\n\n")
            
            for pi in circuit.primary_inputs:
                f.write(f"INPUT({pi})\n")
            for ki in circuit.key_inputs:
                f.write(f"INPUT({ki})\n")
            
            f.write("\n")
            
            for po in circuit.primary_outputs:
                f.write(f"OUTPUT({po})\n")
            
            f.write("\n")
            
            for gate in circuit.gates:
                inputs_str = ', '.join(gate.inputs)
                gate_type_str = gate_type_str_map[gate.gate_type]
                f.write(f"{gate.output} = {gate_type_str}({inputs_str})\n")


def main():

    #make this through cli if i have time
    BENCH_FILE = "cN432.bench"  
    NUM_KEY_BITS_SLL = 8            
    NUM_KEY_BITS_SAR = 10             
    SEED = 42

    print(f"Parsing {BENCH_FILE}...")
    original_circuit = BenchmarkParser.parse_bench(BENCH_FILE)
    print(f"Original circuit: {len(original_circuit.primary_inputs)} inputs, "
          f"{len(original_circuit.gates)} gates, "
          f"{len(original_circuit.primary_outputs)} outputs\n")

    print(f"Applying SARLock+SLL (|K1|={NUM_KEY_BITS_SLL}, |K2|={NUM_KEY_BITS_SAR})...")
    locker = SARLock_SLL(original_circuit, NUM_KEY_BITS_SLL, NUM_KEY_BITS_SAR)
    locked_circuit = locker.lock(seed=SEED)

    gate_overhead = len(locked_circuit.gates) - len(original_circuit.gates)
    overhead_pct  = gate_overhead / len(original_circuit.gates) * 100
    print(f"Locked circuit:  {len(locked_circuit.primary_inputs)} inputs, "
          f"{len(locked_circuit.gates)} gates, "
          f"{len(locked_circuit.primary_outputs)} outputs")
    print(f"Total key bits:  {len(locked_circuit.key_inputs)}")
    print(f"Gate overhead:   +{gate_overhead} gates ({overhead_pct:.1f}%)\n")

    base         = BENCH_FILE.replace(".bench", "")
    output_bench = f"{base}_locked.bench"
    output_key   = f"{base}_correct_key.txt"

    BenchmarkParser.write_bench(locked_circuit, output_bench)
    print(f"Locked circuit saved to: {output_bench}")

    with open(output_key, 'w') as f:
        for key_name in sorted(locker.correct_key.keys()):
            f.write(f"{key_name} = {locker.correct_key[key_name]}\n")
    print(f"Correct key saved to:    {output_key}\n")

    print(f"Theoretical #DIPs to break: 2^{NUM_KEY_BITS_SAR} - 1 = {2**NUM_KEY_BITS_SAR - 1:,}")


if __name__ == "__main__":
    main()