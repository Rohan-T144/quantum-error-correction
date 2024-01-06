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
REPS = 1

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
		# self.circ.id()
		for x in range(self.nqr):
			self.circ.id(self.qr[x])
		self.circ.barrier()
		for x in range(self.nqr-1):
			self.circ.cx(self.qr[x],self.anc[x])
			self.circ.cx(self.qr[x+1],self.anc[x])
		self.circ.measure(self.anc, self.link[i])
		# self.anc.index
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
	p_err = p_meas
	error_meas = pauli_error([('X', p_err), ('I', 1 - p_err)])
	error_gate1 = depolarizing_error(p_err, 1)
	error_gate2 = error_gate1.tensor(error_gate1)

	noise_model = NoiseModel()
	# noise_model.add_all_qubit_quantum_error(error_meas, "id")
	error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
	error_gate1 = depolarizing_error(p_gate, 1)
	error_gate2 = error_gate1.tensor(error_gate1)

	# noise_model = NoiseModel()
	noise_model.add_all_qubit_quantum_error(error_meas, 'id')
	# noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
	# noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
	# noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])  # two qubit gate error is applied to cx gates

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
		print(f'Hz: {self.Hz}')
		self.d = d
		self.match = Matching(
			self.Hz,
			spacelike_weights=np.log((1-p_gate)/p_gate),
			repetitions=REPS,
			timelike_weights=np.log((1-p_meas)/p_meas)
		)
		# Matching.add_noise()
		# self.match.draw()

	def correctreadout(self, repbits, synstrs):
		# print(synstrs)
		synarr = [[] for _ in range(len(synstrs[0]))]
		for s in synstrs:
			l = list(map(int, [*s]))
			for i in range(len(l)):
				synarr[i].append(l[i])
			# synarr.append(l)
		syn = np.array(synarr)
		syn[:,1:] = (syn[:,1:] - syn[:,0:-1]) % 2
		
		# print(syn)
		# syn = np.array(list(map(lambda x: [int(x)], [*anc])))
		# print(syn)
		# print(syn)
		c = self.match.decode(syn)
		readout = np.array(list(map(int, [*repbits])))
		return (c + readout) % 2
		# print(c)
		# for i, v in enumerate(c):
		# 	if v==1:
		# 		# pass
		# 		readout[i] = 1-readout[i]
		# return readout

	def getlogbit(self, readout):
		s = sum(readout)
		if 2*s <= self.d:
			return 0
		else:
			return 1
		# numofbit = [0, 0]
		# for bit in readout:
		# 	numofbit[bit] += 1
		# if numofbit[0] > numofbit[1]:
		# 	return 0
		# else:
		# 	return 1


def testrepwd(d, p_meas, p_gate):
	# d = 5
	qubit = RepQubit(d)
	# qubit.x()
	# qubit.stabilise()
	for i in range(REPS):
		qubit.stabilise(i)
	qubit.circ.measure(qubit.qr, qubit.c)
	print(qubit.draw())
	numshots = 10000

	perror = 0.0
	counts = execute(
		qubit.circ, 
		Aer.get_backend('qasm_simulator'), 
		noise_model=get_noise(p_meas, p_gate), 
		# optimization_level=0,
		shots=numshots
	).result().get_counts()
	print(counts)
	decoder = RepDecoder(qubit.nqr, p_meas, p_gate)
	# s = set()
	for bitstr in counts:
		meas = bitstr.split()

		d = qubit.nqr
	# 	meas = bitstr.split()[0]
	# 	repbits = meas[:d]
	# 	ancbits = meas[d:]
		# res = decoder.correctreadout(meas[-1], meas[:-1])
	# 	# outbit = decoder.getlogbit(res)
		res = list(map(int, [*meas[-1]]))
		outbit = decoder.getlogbit(res)
		# s.add(tuple(res))
	# 	# print(bitstr, counts[bitstr])
	# 	# print(res)
	# 	# print(outbit)
	# 	# print(bitstr, counts[bitstr])
	# 	# print(res)
		if outbit!=0:
			perror += counts[bitstr] / numshots
	# # print(counts)
	# print(s)
	print(perror)
	return perror

testrepwd(5, 0.05, 0.05)

# xvals = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
# xvals = np.arange(0.0, 1.0, 0.1)
# xvals = [.1*i+.05 for i in range(10)]
# yvals = []
# for x in xvals:
# 	yvals.append(testrepwd(5, x, x))
# plt.plot(xvals, yvals)
# # xt = np.arange(0, 1.0, 0.1)
# plt.plot(xvals, xvals, color ='tab:orange')
# plt.show()

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

