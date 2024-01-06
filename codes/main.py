from pprint import pprint

from qiskit import QuantumCircuit, Aer, assemble
from qiskit import QuantumRegister, ClassicalRegister
# from qiskit.ignis.verification.topological_codes import GraphDecoder
# from qiskit.ignis.verification.topological_codes import RepetitionCode
# pip install git+https://github.com/NCCR-SPIN/topological_codes.git
from topological_codes import RepetitionCode
from topological_codes import GraphDecoder
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import pauli_error, depolarizing_error
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram

aer_sim = Aer.get_backend("aer_simulator")


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        error_meas, "measure"
    )  # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(
        error_gate1, ["x"]
    )  # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(
        error_gate2, ["cx"]
    )  # two qubit gate error is applied to cx gates

    return noise_model


def sim_and_plot(qc):
    qc.draw(output="mpl").show()
    qobj = assemble(qc)
    counts = aer_sim.run(qobj).result().get_counts()
    plot_histogram(counts).show()


def _qc_init(cq):
    qc_init = QuantumCircuit(cq)
    qc_init.h(cq[0])
    qc_init.cx(cq[0], cq[1])
    qc_init.x(cq[0])
    return qc_init


def demo():
    cq = QuantumRegister(2, "code_qubit")
    lq = QuantumRegister(1, "auxiliary_qubit")
    sb = ClassicalRegister(1, "syndrome_bit")
    cb = ClassicalRegister(2, "code_bits")

    qc = QuantumCircuit(cq, lq, sb, cb)

    qc.h(cq[0])
    qc.cx(cq[0], cq[1])
    qc.x(cq[0])

    qc.cx(cq[0], lq[0])
    qc.cx(cq[1], lq[0])
    # measures syndrome bits
    qc.measure(lq, sb)
    qc_init = _qc_init(cq)
    # _qc = qc.compose(qc_init, front=True)

    sim_and_plot(qc)

    qc.measure(cq, cb)

    sim_and_plot(qc)


def get_raw_results(code, noise_model=None, shots=None):
    circuits = code.get_circuit_list()
    raw_results = {}
    for log in range(2):
        qobj = assemble(circuits[log], shots=shots)
        job = aer_sim.run(qobj, noise_model=noise_model)
        raw_results[str(log)] = job.result().get_counts(str(log))
    return raw_results


def rep(n=3, T=1, xbasis=False):
    code = RepetitionCode(n, T, xbasis=xbasis)
    mpl = code.circuit["0"].draw(output="mpl")
    mpl.suptitle(f"{xbasis=}")
    mpl.show()
    noise_model = get_noise(0.05, 0.05)
    raw_results = get_raw_results(code, noise_model)
    for log in raw_results:
        print("Logical", log, ":", pprint(raw_results[log]), "\n")

    # table_results = get_results(code, noise_model, shots=10000)
    # P = lookuptable_decoding(table_results, raw_results)
    # print('P =', P)


def graph_decoder(d=4, T=3, p_meas=0.05, p_gate=0.05):
    code = RepetitionCode(d, T)
    decoder = GraphDecoder(code)
    noise_model = get_noise(p_meas, p_gate)
    raw_results = get_raw_results(code, noise_model)

    processed_results = code.process_results(raw_results)
    for log in ["0", "1"]:
        for syndrome_measurements, count in processed_results[log].items():
            # graph = decoder.make_error_graph(syndrome_measurements)
            # assert len(graph) == 1
            # graph = graph["0"]
            # for u, v in graph.edge_list():
            #     edge_info = graph.get_edge_data(u, v)
            #     print('For edge', u, v, 'the weight is', edge_info)

            logical_readout = syndrome_measurements.split("  ")[0]
            corrected_values = decoder.matching(syndrome_measurements)
            print("result", logical_readout, "corrected logical values", corrected_values)


if __name__ == "__main__":
    # rep(xbasis=False)
    # demo()
    graph_decoder()
