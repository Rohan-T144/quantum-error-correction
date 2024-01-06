#!/usr/bin/env python3

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram

from pymatching import Matching
import numpy as np

REPS = 1

def get_noise(p_meas,p_gate):
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    eg = pauli_error([('X',p_gate), ('I', 1 - p_gate)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    # noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ["id"])
    return noise_model


class RepQubit:
	def __init__(self, d = 3, reps = REPS):
		self.qr = QuantumRegister(d, name='qr')
		self.c = ClassicalRegister(d, name='c')
		self.nqr = d
		self.nanc = d-1
		self.anc = QuantumRegister(self.nanc, name='anc')
		self.link = []
		for i in range(reps):
			self.link.append(ClassicalRegister(self.nanc, name=f'link_{i}_qr'))
		# self.syn = ClassicalRegister(self.nanc * reps, name='syn')
		self.circ = QuantumCircuit(self.qr, self.c, self.anc, *self.link)
		self.reps = reps


	def draw(self):
		return self.circ.draw()

	def entangle(self):
		for i in range(1, self.nqr):
			self.circ.cx(self.qr[0], self.qr[i])
		self.circ.barrier()

	def stabilise(self, i = 0):
		self.circ.barrier()
		for x in range(self.nqr):
			self.circ.x(self.qr[x])
		self.circ.barrier()
		for x in range(self.nqr-1):
			self.circ.cx(self.qr[x],self.anc[x])
			self.circ.cx(self.qr[x+1],self.anc[x])
		self.circ.barrier()
		self.circ.measure(self.anc, self.link[i])
		# self.circ.measure(self.anc, self.syn())
		# self.circ.reset(self.anc)
		# self.circ.barrier()
		# self.circ
		# print(self.syn.instances_c)

	def stabrep(self):
		for _ in range(self.reps):
			self.stabilise()

	def x(self):
		for i in range(0,self.nqr):
			self.circ.x(self.qr[i])
		self.circ.barrier()

class RepDecoder:
	def __init__(self, d, p_meas, p_gate):
		hzl = []
		for i in range(d-1):
			row = [0]*d; row[i] = 1; row[i+1] = 1
			hzl.append(row)
		self.Hz = np.array(hzl)
		print(f'Hz: {self.Hz}')
		self.d = d
		self.match = Matching(
			self.Hz,
			# spacelike_weights=np.log((1-p_gate)/p_gate),
			# repetitions=REPS,
			# timelike_weights=np.log((1-p_meas)/p_meas)
		)
		# Matching.add_noise()
		# self.match.draw()

	def correct(self, out, syn):
		c = self.match.decode(syn)
		return (out + c)%2

	def getlogbit(self, readout):
		s = sum(readout)
		if 2*s <= self.d:
			return 0
		else:
			return 1

def testrepwd(d, p_meas, p_gate):
	# d = 5
	qubit = RepQubit(d)
	# qubit.x()
	# qubit.stabilise()
	for i in range(REPS):
		qubit.stabilise(i)
	qubit.circ.measure(qubit.qr, qubit.c)
	# print(qubit.draw())
	numshots = 6000

	perror = 0.0
	counts = execute(
		qubit.circ, 
		Aer.get_backend('qasm_simulator'), 
		noise_model=get_noise(p_meas, p_gate), 
		# optimization_level=0,
		shots=numshots
	).result().get_counts()
	# print(counts)
	decoder = RepDecoder(qubit.nqr, p_meas, p_gate)
	# s = set()
	for bitstr in counts:
		meas = bitstr.split()

		out = np.array(list(map(int, [*meas[-1]])))
		syn = np.array(list(map(int, [*meas[0]])))
		# print(bitstr)
		# print(out, syn, counts[bitstr])
		out = decoder.correct(out, syn)
		outbit = decoder.getlogbit(out)

		if outbit!=1:
			perror += counts[bitstr]
	# print(counts)
	# print(s)
	perror /= numshots
	print(perror)
	return perror

# testrepwd(3, 0.0, 0.05)
xv = np.linspace(0.01, 0.5, 7)
yvs = []
plt.plot(xv, xv)
for i in range(3, 6, 2):
	yv = [testrepwd(i, p, p) for p in xv]
	plt.plot(xv, yv)
plt.show()