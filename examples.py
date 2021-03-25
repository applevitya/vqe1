from qiskit import *

import numpy as np
import matplotlib.pyplot as plt
from math import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import  XGate, IGate, YGate, ZGate
from gradients import U_circuit
from numpy.random import random, multinomial
from vqe import tensordot_krauses

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



def probability121(psi, N):  # probabilities of coincidence
    H = np.array([[1.0], [0.0]])
    V = np.array([[0.0], [1.0]])
    D = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
    A = np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
    R = np.array([[1 / np.sqrt(2)], [1j / np.sqrt(2)]])
    L = np.array([[1 / np.sqrt(2)], [-1j / np.sqrt(2)]])
    if N == 1: #HH
        M = np.dot((np.kron(H, H)),
                   (np.transpose(np.kron(H, H).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 2: #HV
        M = np.dot((np.kron(H, V)),
                   (np.transpose(np.kron(H, V).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 3: #VH
        M = np.dot((np.kron(V, H)),
                   (np.transpose(np.kron(V, H).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 4: #VV
        M = np.dot((np.kron(V, V)),
                   (np.transpose(np.kron(V, V).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 5:  # DD
        M = np.dot((np.kron(D, D)),
                   (np.transpose(np.kron(D, D).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 6:  # DA
        M = np.dot((np.kron(D, A)),
                   (np.transpose(np.kron(D, A).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 7:  # AD
        M = np.dot((np.kron(A, D)),
                   (np.transpose(np.kron(A, D).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 8:  # AA
        M = np.dot((np.kron(A, A)),
                   (np.transpose(np.kron(A, A).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))

    if N == 9:  # RR
        M = np.dot((np.kron(R, R)),
                   (np.transpose(np.kron(R, R).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 10:  # RL
        M = np.dot((np.kron(R, L)),
                   (np.transpose(np.kron(R, L).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 11:  # LR
        M = np.dot((np.kron(L, R)),
                   (np.transpose(np.kron(L, R).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N == 12:  # LL
        M = np.dot((np.kron(L, L)),
                   (np.transpose(np.kron(L, L).conj())))
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))

    return p.real

def probability(psi, N):  # probabilities of coincidence
    H = np.array([[1.0], [0.0]])
    V = np.array([[0.0], [1.0]])
    D = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
    A = np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
    R = np.array([[1 / np.sqrt(2)], [1j / np.sqrt(2)]])
    L = np.array([[1 / np.sqrt(2)], [-1j / np.sqrt(2)]])
    if N == 1: #HH
        M = np.dot((np.kron(H, H)),(np.transpose(np.kron(H, H).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 2: #HV
        M = np.dot((np.kron(H, V)),
                   (np.transpose(np.kron(H, V).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 3: #VH
        M = np.dot((np.kron(V, H)),
                   (np.transpose(np.kron(V, H).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 4: #VV
        M = np.dot((np.kron(V, V)),
                   (np.transpose(np.kron(V, V).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 5:  # DD
        M = np.dot((np.kron(D, D)),
                   (np.transpose(np.kron(D, D).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 6:  # DA
        M = np.dot((np.kron(D, A)),
                   (np.transpose(np.kron(D, A).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 7:  # AD
        M = np.dot((np.kron(A, D)),
                   (np.transpose(np.kron(A, D).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 8:  # AA
        M = np.dot((np.kron(A, A)),
                   (np.transpose(np.kron(A, A).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))

    if N == 9:  # RR
        M = np.dot((np.kron(R, R)),
                   (np.transpose(np.kron(R, R).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 10:  # RL
        M = np.dot((np.kron(R, L)),
                   (np.transpose(np.kron(R, L).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 11:  # LR
        M = np.dot((np.kron(L, R)),
                   (np.transpose(np.kron(L, R).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))
    if N == 12:  # LL
        M = np.dot((np.kron(L, L)),
                   (np.transpose(np.kron(L, L).conj())))
        p = np.dot(psi.conjugate().transpose(),(np.dot((np.dot(M.conjugate().transpose(),M)),psi)))

    return p.real[0][0]



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
    psi = U_circuit(phi,0).data @ tensordot_krauses(state_zero.data,state_zero.data)
    II = probabilityRandom_HV(psi,0,n)+probabilityRandom_HV(psi,1,n)+probabilityRandom_HV(psi,2,n)+probabilityRandom_HV(psi,3,n)
    XX = probabilityRandom_DA(psi,0,n)-probabilityRandom_DA(psi,1,n)-probabilityRandom_DA(psi,2,n)+probabilityRandom_DA(psi,3,n)
    YY = probabilityRandom_RL(psi,0,n)-probabilityRandom_RL(psi,1,n)-probabilityRandom_RL(psi,2,n)+probabilityRandom_RL(psi,3,n)
    ZZ = probabilityRandom_HV(psi,0,n)-probabilityRandom_HV(psi,1,n)-probabilityRandom_HV(psi,2,n)+probabilityRandom_HV(psi,3,n)
    ZI = probabilityRandom_HV(psi,0,n)+probabilityRandom_HV(psi,1,n)-probabilityRandom_HV(psi,2,n)-probabilityRandom_HV(psi,3,n)
    return (II+XX+YY+1/2*(ZZ-ZI))/n

phi = [1,1,1,1,1,1]
print('istina', energy(phi))
print('fin stat', schwinger_samples(phi, 10000))