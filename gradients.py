from qiskit import *
from qiskit.visualization import plot_histogram, plot_state_city
from math import *
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate, CCXGate
import numpy as np

from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

def schwinger_matrix(m, k): #k  #1+2XX+2YY+0.5(-ZI+ZZ+mIZ-mZI)
    I = Operator(IGate())
    X = Operator(XGate())
    Y = Operator(YGate())
    Z = Operator(ZGate())
    if k == 1:
        return I.expand(I)
    if k == 2:
        return X.expand(X)
    if k == 3:
        return Y.expand(Y)
    if k == 5:
        return Z.expand(I)
    if k == 4:
        return Z.expand(Z)
    if k == 6:
        return I.expand(Z)
    if k == 7:
        return Z.expand(I)
#I.expand(I)+2*X.expand(X)+2*Y.expand(Y)+0.5*(-Z.expand(I)+Z.expand(Z)+m*I.expand(Z)-m*Z.expand(I))




def U(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return ((Y.conjugate().transpose()).compose(Z.conjugate().compose(Y)))


def waveplate(phi, delta):
    """Return waveplate matrix with retardance delta and axis angle phi.

    delta = pi for HWP
    delta = pi/2 for QWP
    """
    T = cos(delta / 2) + 1j * sin(delta / 2) * cos(2 * phi)
    R = 1j * sin(delta / 2) * sin(2 * phi)
    return Operator([[T, R], [-R.conjugate(), T.conjugate()]])

def derivative_U1121212(phi, delta):
    T =  - 2j * sin(delta / 2) * sin(2 * phi)
    R = 2j * sin(delta / 2) * cos(2 * phi)
    return Operator([[T, R], [-R.conjugate(), T.conjugate()]])

def derivative_U(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))

    return  Operator(RYGate(2 * phi + pi)).compose(Z.conjugate(), front=True).compose(Y.transpose(),front=True)

def U_circuit(phi, N):
    if N == 0:
        return ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()))).compose(
            Operator(CXGate())).compose((U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(
            Operator(XGate()))).compose(Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 2:
        return (((U(phi[0], pi / 2).compose(derivative_U(phi[1], pi))).expand(
            Operator(XGate()))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 3:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()))).compose(
            Operator(CXGate()))).compose(
            (derivative_U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 4:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(derivative_U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 5:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((derivative_U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 6:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(derivative_U(phi[5], pi)))))


def B(phi, N, k):

    return (U_circuit(phi, 0).conjugate().transpose()).compose(schwinger_matrix(0,k).compose(U_circuit(phi, N),front= True),front=True)


def C_Gate(B):  # n-number of qubits
    B=B.data
    x, y = 8, 8
    C = [[0 for j in range(y)] for i in range(x)]
    C = np.empty((8,8), dtype="object")
    for i in range(0, 8):
        for j in range(0, 8):
            if i == j:
                C[i][j] = 1
            else:
                C[i][j] = 0

    for m in range(4,8):
        for n in range(4,8):
            C[n][m] = B[n-4][m-4]
    return Operator(C)

def C_Gate_new(B, n):  # n-number of qubits
    B=B.data
    x, y = 8, 8
    C = [[0 for j in range(y)] for i in range(x)]
    C = np.empty((8,8), dtype="object")
    for i in range(0, 8):
        for j in range(0, 8):
            if i == j:
                C[i][j] = 1
            else:
                C[i][j] = 0

    for m in range(0,4):
        for n in range(0,4):
            C[n][m] = B[n][m]
    return Operator(C)



#################################circuit -- 2##########################################################
def derivative_U2(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return -Y.compose(Z.conjugate(), front=True).compose(Operator(RYGate(-2 * phi + pi)).conjugate(), front= True)

def U_circuit2(phi, N):
    if N == 0:
        return ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate())).compose((U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U2(phi[0], pi / 2).compose(U(phi[1], pi))).expand(
            Operator(XGate()).compose(Operator(IGate())))).compose(Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 2:
        return (((U(phi[0], pi / 2).compose(derivative_U2(phi[1], pi))).expand(
            Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 3:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (derivative_U2(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 4:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(derivative_U2(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 5:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((derivative_U2(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 6:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(derivative_U2(phi[5], pi)))))

def B2(phi, N, k):

    return (U_circuit2(phi, 0).conjugate().transpose()).compose(schwinger_matrix(0,k).compose(U_circuit2(phi, N),front= True),front= True)

##########################################

SH = 100
def hadamard(phi,N,k):
    qc = QuantumCircuit(3, 1)
    qc.h(2)
    qc.append(C_Gate(B(phi,N,k)),[0,1,2])
    qc.h(2)
    qc.measure(2, 0)
    #qc.draw('mpl').show()

    qc2 = QuantumCircuit(3, 1)
    qc2.h(2)
    qc2.append(C_Gate(B2(phi, N, k)), [0, 1, 2])
    qc2.h(2)
    qc2.measure(2, 0)

    backend = BasicAer.get_backend('qasm_simulator')
    backend_2 = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=SH)
    job2 = execute(qc2, backend_2,shots=SH)
    total = job.result().get_counts(qc)['0']
    total2 = job2.result().get_counts(qc2)['0']
    #plt = plot_histogram(job.result().get_counts(qc), color='midnightblue', title="New Histogram")
    #qc.draw('mpl').show()
    #plt.show()
    return 4*total/SH+4*total2/SH-4


def hadamard121(phi,N,k):
    qc = QuantumCircuit(3, 1)
    qc.h(2)
    qc.append(C_Gate(B(phi, N, k)),[0,1,2])
    qc.h(2)
    #qc.draw('mpl').show()


    qc2 = QuantumCircuit(3, 1)
    qc2.h(2)
    qc2.append(C_Gate(B2(phi, N, k)), [0, 1, 2])
    qc2.h(2)

    simulation = Aer.get_backend('statevector_simulator')
    stat_vector = execute(qc, simulation).result().get_statevector(qc)
    stat_vector_2  = execute(qc2, simulation).result().get_statevector(qc2)

    total_1 = np.sum((np.abs(stat_vector)**2)[:4])
    total_2 = np.sum((np.abs(stat_vector_2)**2)[:4])
    #total_1 = Statevector(stat_vector).probabilities([2])[0]
    #total_2 = Statevector(stat_vector_2).probabilities([2])[0]

    return  (4*total_1+4*total_2-4)




def hadamard_test(phi, N):
    return hadamard(phi,N,1)+hadamard(phi,N,2)+hadamard(phi,N,3)+0.5*hadamard(phi,N,4)-0.5*hadamard(phi,N,5)+0.5*0*hadamard(phi,N,6)-0.5*0*hadamard(phi,N,7)





