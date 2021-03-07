from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer
from math import *
from qiskit.aqua.operators import One
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate, CCXGate
from gradients import U_circuit, U_circuit2

############################################################
I = Operator(IGate())
X = Operator(XGate())
Y = Operator(YGate())
Z = Operator(ZGate())

state_zero = Operator(np.array([[1.0],[0.0]]))
def schwinger(m):
    return I.expand(I)+2*X.expand(X)+2*Y.expand(Y)+0.5*(-Z.expand(I)+Z.expand(Z)+m*I.expand(Z)-m*Z.expand(I))

############################################################

phi = [1,2,2,3,2,2]
def energy(phi):
    psi = (state_zero ^ state_zero) @ U_circuit(phi, 0)
    energy = (psi @ schwinger(0)) @ ((psi).transpose())
    return energy.data[0][0]

def der(phi,d,N): #d = delta(argument)
    a = energy(phi)
    phi_d = phi
    phi_d[N] = phi_d[N]+d                    # phi+d_phi
    return ((energy(phi_d)-a)/d).real


def plot_graph(der,phi,N):
    xrange=np.arange(0,0.2,0.005)
    yrange=[der(phi,i,N) for i in xrange]
    plt.plot(xrange,yrange)
    plt.xlabel('delta theta')
    plt.ylabel('mean of gradient')
    plt.show()

plot_graph(der,phi,1)