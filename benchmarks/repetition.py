import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram



class RepQubit:
	def __init__(self, d = 3):
		self.qr = QuantumRegister(d, name='qr')
		self.cr = ClassicalRegister(d, name='cr')
		self.nqr = d
		self.nanc = d-1
		self.anc = QuantumRegister(d-1, name='anc')
		self.syn = ClassicalRegister(d-1, name='syn')
		self.circ = QuantumCircuit(self.qr, self.cr, self.anc, self.syn)

	def draw(self):
		return self.circ.draw()

	def stabilise(self):
		for i in range(self.nqr-1):
			self.circ.cx(self.qr[i],self.anc[i])
			self.circ.cx(self.qr[i+1],self.anc[i])
		self.circ.measure(self.anc, self.syn)

		self.circ.reset(self.anc)
		self.circ.barrier()

	def x(self):
		for i in range(0,self.nqr):
			self.circ.x(self.qr[i])
		self.circ.barrier()

class Decoder:
	def __init__(self, d = 3):
		self.nqr = d; self.nanc = d-1



