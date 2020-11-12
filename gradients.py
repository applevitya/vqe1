from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate


def U(phi, delta):
    Y = Operator(RYGate(2*phi))
    Z = Operator(RZGate(delta))

    return (Y.compose(Z.compose(Y.transpose())))


print(U(pi/8, pi/8).data)
print(waveplate(pi/8, pi/8))


