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
from numpy.random import random, multinomial
from qutip import tensor, Qobj

def getQobj(psi):
    return Qobj(psi)
############################################################
I = Operator(IGate())
X = Operator(XGate())
Y = Operator(YGate())
Z = Operator(ZGate())

state_zero = Operator(np.array([[1.0], [0.0]]))


def schwinger(m):
    return I.expand(I) + X.expand(X) + Y.expand(Y) + 0.5 * (
                -Z.expand(I) + Z.expand(Z) + m * I.expand(Z) - m * Z.expand(I))


############################################################

phi = [1, 1, 2, 2, 2, 2]


def energy(phi):
    psi = (state_zero ^ state_zero) @ U_circuit(phi, 0)
    energy = (psi @ schwinger(0)) @ ((psi).conjugate().transpose())
    return energy.data[0][0]


def der(phi, d, N):  # d = delta(argument)
    N = N - 1
    a = energy(phi)
    phi_d = phi
    phi_d[N] = phi_d[N] + d  # phi+d_phi
    return ((energy(phi_d) - a) / d)


def plot_graph(der, phi, N):
    xrange = np.arange(0, 0.00000001, 0.000000001)
    yrange = [der(phi, i, N) for i in xrange]
    plt.plot(xrange, yrange)
    plt.xlabel('delta theta')
    plt.ylabel('mean of gradient')
    plt.show()


#########################################################################################################################



def probability(psi, N):  # probabilities of coincidence
    H = np.array([[1.0], [0.0]])
    V = np.array([[0.0], [1.0]])
    D = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
    A = np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
    R = np.array([[1 / np.sqrt(2)], [1j / np.sqrt(2)]])
    L = np.array([[1 / np.sqrt(2)], [-1j / np.sqrt(2)]])
    if N == 1: #HH
        M = np.dot((tensor(getQobj(H), getQobj(H))),
                   (np.transpose(tensor(getQobj(H), getQobj(H)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 2: #HV
        M = np.dot((tensor(getQobj(H), getQobj(V))),
                   (np.transpose(tensor(getQobj(H), getQobj(V)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 3: #VH
        M = np.dot((tensor(getQobj(V), getQobj(H))),
                   (np.transpose(tensor(getQobj(V), getQobj(H)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 4: #VV
        M = np.dot((tensor(getQobj(V), getQobj(V))),
                   (np.transpose(tensor(getQobj(V), getQobj(V)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 5:  # DD
        M = np.dot((tensor(getQobj(D), getQobj(D))),
                   (np.transpose(tensor(getQobj(D), getQobj(D)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 6:  # DA
        M = np.dot((tensor(getQobj(D), getQobj(A))),
                   (np.transpose(tensor(getQobj(D), getQobj(A)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 7:  # AD
        M = np.dot((tensor(getQobj(A), getQobj(D))),
                   (np.transpose(tensor(getQobj(A), getQobj(D)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 8:  # AA
        M = np.dot((tensor(getQobj(A), getQobj(A))),
                   (np.transpose(tensor(getQobj(A), getQobj(A)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))

    if N == 9:  # RR
        M = np.dot((tensor(getQobj(R), getQobj(R))),
                   (np.transpose(tensor(getQobj(R), getQobj(R)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 10:  # RL
        M = np.dot((tensor(getQobj(R), getQobj(L))),
                   (np.transpose(tensor(getQobj(R), getQobj(L)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 11:  # LR
        M = np.dot((tensor(getQobj(L), getQobj(R))),
                   (np.transpose(tensor(getQobj(L), getQobj(R)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 12:  # LL
        M = np.dot((tensor(getQobj(L), getQobj(L))),
                   (np.transpose(tensor(getQobj(L), getQobj(L)).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))

    return p.real

def probabilityRandom_HV(psi, N, n):  # n-value of samples
    probabilityArray = np.random.multinomial(n, [probability(psi, 1), probability(psi, 2), probability(psi, 3),
                                                 probability(psi, 4)], 1)
    return probabilityArray[0][N]

def probabilityRandom_DA(psi, N, n):
    probabilityArray = np.random.multinomial(n, [probability(psi, 5), probability(psi, 6), probability(psi, 7),
                                                 probability(psi, 8)], 1)
    return probabilityArray[0][N]

def probabilityRandom_RL(psi, N, n):
    probabilityArray = np.random.multinomial(n, [probability(psi, 9), probability(psi, 10), probability(psi, 11),
                                                 probability(psi, 12)], 1)
    return probabilityArray[0][N]


def schwinger_samples(phi,n):
    psi = np.dot(U_circuit(phi,0).data,tensor(getQobj(state_zero.data), getQobj(state_zero.data)))
    II = probabilityRandom_HV(psi,0,n)+probabilityRandom_HV(psi,1,n)+probabilityRandom_HV(psi,2,n)+probabilityRandom_HV(psi,3,n)
    XX = probabilityRandom_DA(psi,0,n)-probabilityRandom_DA(psi,1,n)-probabilityRandom_DA(psi,2,n)+probabilityRandom_DA(psi,3,n)
    YY = probabilityRandom_RL(psi,0,n)-probabilityRandom_RL(psi,1,n)-probabilityRandom_RL(psi,2,n)+probabilityRandom_RL(psi,3,n)
    ZZ = probabilityRandom_HV(psi,0,n)-probabilityRandom_HV(psi,1,n)-probabilityRandom_HV(psi,2,n)+probabilityRandom_HV(psi,3,n)
    ZI = probabilityRandom_HV(psi,0,n)+probabilityRandom_HV(psi,1,n)-probabilityRandom_HV(psi,2,n)-probabilityRandom_HV(psi,3,n)
    return (II+XX+YY+1/2*(ZZ-ZI))/n