#!/usr/bin/env python
# coding: utf-8

# # Repetition Code Testing
# 

# In[7]:


import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.visualization import plot_histogram

# import matplotlib.pyplot as plt

# from qtcodes import RepetitionQubit

# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
# from qiskit.visualization import plot_histogram


# In[4]:



class RepQubit:
	def __init__(self, d = 3):
		self.qr = QuantumRegister(d, name='qr')
		self.c = ClassicalRegister(d, name='c')
		self.nqr = d
		self.nanc = d-1
		self.anc = QuantumRegister(d-1, name='anc')
		self.syn = ClassicalRegister(d-1, name='syn')
		self.circ = QuantumCircuit(self.qr, self.c, self.anc, self.syn)

	def draw(self):
		return self.circ.draw()

	def entangle(self):
		for i in range(1, self.nqr):
			self.circ.cx(self.qr[0], self.qr[i])
		self.circ.barrier()

	def stabilise(self):
		for i in range(self.nqr-1):
			self.circ.cx(self.qr[i],self.anc[i])
			self.circ.cx(self.qr[i+1],self.anc[i])
		self.circ.measure(self.anc, self.syn)

		self.circ.reset(self.anc)
		self.circ.barrier()
		# self.circ
		# print(self.syn.instances_c)

	def x(self):
		for i in range(0,self.nqr):
			self.circ.x(self.qr[i])
		self.circ.barrier()


# In[10]:


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])  # two qubit gate error is applied to cx gates

    return noise_model


# In[59]:


# nd = 5
# qubit = RepQubit(nd)
# # qubit.x()
# qubit.stabilise()

# qubit.circ.measure_all()

# qubit.draw()


# In[8]:


from pymatching import Matching
import numpy as np

class RepDecoder:
    def __init__(self, d = 3):
        hzl = []
        for i in range(d-1):
            row = [0]*d; row[i] = 1; row[i+1] = 1
            hzl.append(row)
        self.Hz = np.array(hzl)
        # print(self.Hz)
        self.d = d
        self.match = Matching(self.Hz)
        # self.match.draw()

    def correctreadout(self, repbits, anc):
        syn = np.array(list(map(int, [*anc])))
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
    
    


# In[70]:


# numshots = 10000
# perror = 0.0
# counts = execute(qubit.circ, Aer.get_backend('qasm_simulator'), noise_model=get_noise(0.1,0.1), shots=numshots).result().get_counts()
# decoder = RepDecoder(qubit.nqr)
# for bitstr in counts:
#     d = qubit.nqr
#     meas = bitstr.split()[0]
#     repbits = meas[:d]
#     ancbits = meas[d:]
#     res = decoder.correctreadout(repbits, ancbits)
#     outbit = decoder.getlogbit(res)
#     # print(bitstr, counts[bitstr])
#     # print(res)
#     # print(outbit)
#     if outbit!=0:
#         perror += counts[bitstr] / numshots
# # print(counts)
# print(perror)


# In[12]:


def testrep(d, p_meas, p_gate):
    # d = 5
    qubit = RepQubit(d)
    # qubit.x()
    qubit.stabilise()
    qubit.circ.measure_all()
    # print(qubit.draw())
    numshots = 10000

    perror = 0.0
    counts = execute(qubit.circ, Aer.get_backend('qasm_simulator'), noise_model=get_noise(p_meas, p_gate), shots=numshots).result().get_counts()
    # print(counts)
    decoder = RepDecoder(qubit.nqr)
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

testrep(5, 0.05, 0.05)


# In[140]:


# testrep(3, 0.1)
xvals = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
yvals = []
for x in xvals:
    yvals.append(testrep(3, x, x))
plt.plot(xvals, yvals)
xt = np.arange(0, 0.2, 0.02)
plt.plot(xt, xt, color ='tab:orange')
plt.show()


# In[147]:


xv = range(3, 12, 2)
yv = []
from math import pi
for x in xv:
    yv.append(testrep(x,pi/20, pi/10))
plt.plot(xv, yv)
# plt.plot(xv, [pi/10]*len(yv))
plt.show()

# In[15]:


def testrepwithdecode(d, p_meas, p_gate):
    # d = 5
    qubit = RepQubit(d)
    # qubit.x()
    qubit.stabilise()
    qubit.circ.measure_all()
    # print(qubit.draw())
    numshots = 10000

    perror = 0.0
    counts = execute(qubit.circ, Aer.get_backend('qasm_simulator'), noise_model=get_noise(p_meas, p_gate), shots=numshots).result().get_counts()
    decoder = RepDecoder(qubit.nqr)
    cnts = []
    for bitstr in counts:
        d = qubit.nqr
        meas = bitstr.split()[0]
        repbits = meas[:d]
        ancbits = meas[d:]
        # ancbits = bitstr.split()[1]
        res = decoder.correctreadout(repbits, ancbits)
        outbit = decoder.getlogbit(res)
        # outbit = decoder.getlogbit(list(map(int, [*repbits])))
        # print(bitstr, counts[bitstr])
        # print(res)
        # print(outbit)
        # print(bitstr, counts[bitstr])
        # print(res)
        # cnts.append((counts[bitstr], res, bitstr))
        if outbit!=0:
            perror += counts[bitstr] / numshots
    # print(counts)
    # for x in reversed(sorted(cnts)):
    #     print(x)
    print(perror)
    return perror


# In[165]:


# testrepwithdecode(3, 0, pi/10)


# In[16]:


xv = range(3, 12, 2)
yv = []
from math import pi
for x in xv:
    yv.append(testrepwithdecode(x, pi/20, pi/10))
plt.plot(xv, yv)


# In[20]:


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
    # print(counts)
    decoder = RepDecoder(qubit.nqr)
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

testrepwd(5, 0.05, 0.05)

xv = range(3, 12, 2)
yv = []
from math import pi
for x in xv:
    yv.append(testrepwd(x, pi/20, pi/10))
plt.plot(xv, yv)
plt.show()

