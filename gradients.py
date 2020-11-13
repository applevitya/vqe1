from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate

Y = Operator(RYGate(2*phi))
Z = Operator(RZGate(delta))

def U(phi, delta):
    return (Y.compose(Z.compose(Y.transpose())))


print(U(pi/8, pi/8).data)
print(waveplate(pi/8, pi/8))

def plate(t, d):  # delta, theta
    T = cos(d/2) + 1j * sin(d/2) * cos(2 * t)
    R = 1j * sin(d/2) * sin(2 * t)
    D = np.array([[T, R],
                  [-R.conjugate(), T.conjugate()]])
    return D

print(plate(pi/8,pi/8))

