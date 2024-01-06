# Benchmarking Quantum Error Correction Codes

This repository provides code used in my research project that compared and analysed implementations of the repetition, shor and surface error correction schemes. The results are explained in the [paper](https://www.jsr.org/hs/index.php/path/article/view/5127) published by the Journal of Student Research

## Abstract

Quantum Computing is a computing framework that takes advantage of unique quantum mechanical properties (such as superposition and entanglement) to perform calculations and implement algorithms that could offer exponential speed-ups over classical computing. However, in physical implementations of such quantum computers, qubits – the fundamental components of these systems – can accumulate errors that must be accounted for. In order to mitigate these errors, various quantum error correction (QEC) codes have been developed, including the repetition code and surface codes. In this experiment I implement and evaluate three types of QEC codes on the Qiskit simulator to compare their efficacy and applicability in correcting for different kinds of errors. I hypothesized that surface codes, with their more effective design and range of correction methods, should perform the best with much lower error thresholds and resultant logical error rates. The results support the hypothesis and suggest that surface codes are a viable method of implementing scalable error correction in quantum computers.

