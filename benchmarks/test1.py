from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise_model(p_err):
	error_gate1 = pauli_error([("X", p_err), ("I", 1 - p_err)])
	noise_model = NoiseModel()
	error_gate1 = depolarizing_error(p_err, 1)
	error_gate2 = error_gate1.tensor(error_gate1)

	noise_model.add_all_qubit_quantum_error(error_gate1, "id")
	# noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
	noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
	noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])
	return noise_model

from qtcodes import TopologicalBenchmark, TopologicalAnalysis
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qtcodes import RepetitionDecoder
from qtcodes import RepetitionQubit
from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister, Aer
from tqdm import tqdm

import multiprocessing as mp

d = 7
T = 1

qubit = RepetitionQubit({'d':d})
qubit.reset_z()
qubit.stabilize()
qubit.id_data()
qubit.stabilize()
qubit.readout_z()
# qubit.draw(output='mpl', fold=50)


tool = TopologicalBenchmark(
	decoder=RepetitionDecoder({"d":d, "T":T}),
	circ=qubit.circ,
	noise_model_func=get_noise_model,
	correct_logical_value = 0
)

print("\nSIMULATE: (d={},T={})\n".format(tool.decoder.params["d"], tool.decoder.params["T"]))
physical_error_rates = [.1*i+.05 for i in range(10)]
tool.sweep(physical_error_rates=physical_error_rates)
print("Done!")

import matplotlib.pyplot as plt
import numpy as np

analysis = TopologicalAnalysis()
for log_plot in [True, False]:
	fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
	ax = fig.subplots()
	print(tool.decoder.params)
	analysis.params["d"] = int(tool.decoder.params["d"][0])
	analysis.params["T"] = int(tool.decoder.params["T"])
	analysis.data["physical_error_rates"] = tool.data["physical_error_rates"]
	analysis.data["logical_error_rates"] = tool.data["logical_error_rates"]
	analysis.plot(
		fig=fig,
		ax=ax,
		label="d={},T={}".format(tool.decoder.params["d"], tool.decoder.params["T"]),
		log=log_plot,
	)
	plt.plot(
		tool.data["physical_error_rates"],
		tool.data["physical_error_rates"],
		"--",
		label="breakeven",
	)
	plt.legend(loc="lower right", prop={"size": 6})
	ax.set_xlabel("Physical Error Rate", size=10)
	ax.set_ylabel("Logical Error Rate", size=10)
	ax.set_title("Surface Code Performance", size=10)
	fig.tight_layout()
	plt.show()
	# fig.save
	# if not log_plot:
	# 	fig.savefig(f'err_prob_d{d}')
	# fig.savefig('test1')