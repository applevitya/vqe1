from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate


def U(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return ((Y.transpose()).compose(Z.conjugate().compose(Y)))

def waveplate(phi, delta):
    """Return waveplate matrix with retardance delta and axis angle phi.

    delta = pi for HWP
    delta = pi/2 for QWP
    """
    T = cos(delta / 2) + 1j * sin(delta / 2) * cos(2 * phi)
    R = 1j * sin(delta / 2) * sin(2 * phi)
    return Operator([[T, R], [-R.conjugate(), T.conjugate()]])




