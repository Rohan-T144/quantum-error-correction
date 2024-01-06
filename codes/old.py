#!/usr/bin/env python

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
# from qiskit.visualization import plot_histogram


class RepQubit:
	def __init__(self, d = 3, reps = 2):
		self.qr = QuantumRegister(d, name='qr')
		self.c = ClassicalRegister(d, name='c')
		self.nqr = d
		self.nanc = d-1
		self.anc = QuantumRegister(self.nanc * reps, name='anc')
		self.syn = ClassicalRegister(self.nanc * reps, name='syn')
		self.circ = QuantumCircuit(self.qr, self.c, self.anc, self.syn)
		self.reps = reps


	def draw(self):
		return self.circ.draw()

	def entangle(self):
		for i in range(1, self.nqr):
			self.circ.cx(self.qr[0], self.qr[i])
		self.circ.barrier()

	def stabilise(self, i = 0):
		for i in range(self.nqr-1):
			self.circ.cx(self.qr[i],self.anc[i])
			self.circ.cx(self.qr[i+1],self.anc[i])
		self.circ.measure(self.anc, self.syn)
		self.anc.index
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


def get_noise(p_meas, p_gate):
	error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
	error_gate1 = depolarizing_error(p_gate, 1)
	error_gate2 = error_gate1.tensor(error_gate1)

	noise_model = NoiseModel()
	noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
	noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
	noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])  # two qubit gate error is applied to cx gates

	return noise_model


from pymatching import Matching
import numpy as np

class RepDecoder:
	def __init__(self, d, p_meas, p_gate):
		hzl = []
		for i in range(d-1):
			row = [0]*d; row[i] = 1; row[i+1] = 1
			hzl.append(row)
		self.Hz = np.array(hzl)
		# print(self.Hz)
		self.d = d
		self.match = Matching(
			self.Hz,
			spacelike_weights=np.log((1-p_gate)/p_gate),
			repetitions=1,
			timelike_weights=np.log((1-p_meas)/p_meas)
		)
		# self.match.draw()

	def correctreadout(self, repbits, anc):
		syn = np.array(list(map(int, [*anc])))
		print(syn)
		# print(syn)
		c = self.match.decode(syn)
		readout = list(map(int, [*repbits]))
		# print(c)
		for i, v in enumerate(c):
			if v==1:
				# pass
				readout[i] = 1-readout[i]
		return readout

	def getlogbit(self, readout):
		numofbit = [0, 0]
		for bit in readout:
			numofbit[bit] += 1
		if numofbit[0] > numofbit[1]:
			return 0
		else:
			return 1
	

def testrep(d, p_meas, p_gate):
	# d = 5
	qubit = RepQubit(d)
	# qubit.x()
	qubit.stabrep()
	qubit.circ.measure_all()
	# print(qubit.draw())
	numshots = 10000

	perror = 0.0
	counts = execute(qubit.circ, Aer.get_backend('qasm_simulator'), noise_model=get_noise(p_meas, p_gate), shots=numshots).result().get_counts()
	# print(counts)
	decoder = RepDecoder(qubit.nqr, p_meas, p_gate)
	for bitstr in counts:
		d = qubit.nqr
		meas = bitstr.split()[0]
		repbits = meas[:d]
		ancbits = meas[d:]
		# res = decoder.correctreadout(repbits, ancbits)
		# outbit = decoder.getlogbit(res)
		outbit = decoder.getlogbit(list(map(int, [*repbits])))
		# print(bitstr, counts[bitstr])
		# print(res)
		# print(outbit)
		# print(bitstr, counts[bitstr])
		# print(res)
		if outbit!=0:
			perror += counts[bitstr] / numshots
	# print(counts)
	print(perror)
	return perror

# testrep(5, 0.05, 0.05)



# # testrep(3, 0.1)
# xvals = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
# yvals = []
# for x in xvals:
#     yvals.append(testrep(3, x, x))
# plt.plot(xvals, yvals)
# xt = np.arange(0, 0.2, 0.02)
# plt.plot(xt, xt, color ='tab:orange')
# plt.show()


# xv = range(3, 12, 2)
# yv = []
# from math import pi
# for x in xv:
#     yv.append(testrep(x,pi/20, pi/10))
# plt.plot(xv, yv)
# # plt.plot(xv, [pi/10]*len(yv))
# plt.show()


def testrepwd(d, p_meas, p_gate):
	# d = 5
	qubit = RepQubit(d)
	# qubit.x()
	qubit.stabilise()
	qubit.circ.measure_all()
	# print(qubit.draw())
	numshots = 10000

	perror = 0.0
	counts = execute(qubit.circ, Aer.get_backend('qasm_simulator'), noise_model=get_noise(p_meas, p_gate), shots=numshots).result().get_counts()
	print(counts)
	decoder = RepDecoder(qubit.nqr, p_meas, p_gate)
	for bitstr in counts:
		d = qubit.nqr
		meas = bitstr.split()[0]
		repbits = meas[:d]
		ancbits = meas[d:]
		res = decoder.correctreadout(repbits, ancbits)
		# outbit = decoder.getlogbit(res)
		# res = list(map(int, [*repbits]))
		outbit = decoder.getlogbit(res)
		# print(bitstr, counts[bitstr])
		# print(res)
		# print(outbit)
		# print(bitstr, counts[bitstr])
		# print(res)
		if outbit!=0:
			perror += counts[bitstr] / numshots
	# print(counts)
	print(perror)
	return perror

testrepwd(3, 0.05, 0.05)

# xv = range(3, 12, 2)
# yv = []
# from math import pi
# for x in xv:
# 	yv.append(testrepwd(x, 0.05, 0.05))
# plt.plot(xv, yv)
# plt.show()


# xv = range(3, 12, 2)
# yv = []
# from math import pi
# for x in xv:
#     yv.append(testrep(x,pi/20, pi/10))
# plt.plot(xv, yv)
# # plt.plot(xv, [pi/10]*len(yv))
# plt.show()

