from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate
import numpy as np


def schwinger_matrix(m):
    I = Operator(IGate())
    X = Operator(XGate())
    Y = Operator(YGate())
    Z = Operator(ZGate())
    return X.expand(X)+I.expand(I)
    #I.expand(I)+2*X.expand(X)+2*Y.expand(Y)+0.5*(-Z.expand(I)+Z.expand(Z)+m*I.expand(Z)-m*Z.expand(I))  #1+2XX+2YY+0.5(-ZI+ZZ+mIZ-mZI)



def waveplate(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return ((Y.transpose()).compose(Z.conjugate().compose(Y)))


def U(phi, delta):
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

    return  Operator(RYGate(2 * phi + pi/2)).compose(Z.conjugate(), front=True).compose(Y.transpose(),front=True)

def U_circuit(phi, N):
    if N == 0:
        return ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate())).compose((U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(
            Operator(XGate()).compose(Operator(IGate())))).compose(Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 2:
        return (((U(phi[0], pi / 2).compose(derivative_U(phi[1], pi))).expand(
            Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 3:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (derivative_U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 4:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(derivative_U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 5:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((derivative_U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 6:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).expand(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).expand((U(phi[4], pi / 2).compose(derivative_U(phi[5], pi)))))


def B(phi, N):

    return (U_circuit(phi, 0).conjugate().transpose()).compose(schwinger_matrix(0).compose(U_circuit(phi, N),front= True),front=True)


def C_Gate(B, n):  # n-number of qubits
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



################################################################################################
def derivative_U2(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return -Y.compose(Z.conjugate(), front=True).compose(Operator(RYGate(-2 * phi + pi / 2)).conjugate(), front= True)

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

def B2(phi, N):

    return (U_circuit2(phi, 0).conjugate().transpose()).compose(schwinger_matrix(0).compose(U_circuit2(phi, N),front= True),front= True)






phi = [1, 1, 1,2 , 1, 3]

print(B(phi,3).is_unitary())
print(schwinger_matrix(0).is_unitary())


def hadamard_test(phi,N):
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.append(C_Gate(B(phi,N),3),[0,1,2])
    qc.h(0)
    qc.measure(0, 0)
    #qc.draw('mpl').show()

    qc2 = QuantumCircuit(3, 1)
    qc2.h(0)
    qc2.append(C_Gate(B2(phi, N), 3), [0, 1, 2])
    qc2.h(0)
    qc2.measure(0, 0)




    backend = BasicAer.get_backend('qasm_simulator')
    backend_2 = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=10000)
    job2 = execute(qc2, backend_2,shots=10000)
    total = job.result().get_counts(qc)['0']
    total2 = job2.result().get_counts(qc2)['0']


    #plt = plot_histogram(job.result().get_counts(qc), color='midnightblue', title="New Histogram")
    #qc.draw('mpl').show()
    #plt.show()

    return 8*total/10000+8*total2/10000-8

def hadamard_test1221(phi,N):
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.append(C_Gate(B(phi, N), 3), [0, 1, 2])
    qc.h(0)

    qc2 = QuantumCircuit(3, 1)
    qc2.h(0)
    qc2.append(C_Gate(B2(phi, N), 3), [0, 1, 2])
    qc2.h(0)

    simulation = Aer.get_backend('statevector_simulator')
    stat_vector = execute(qc, simulation).result().get_statevector(qc)
    stat_vector_2  = execute(qc2, simulation).result().get_statevector(qc2)
    total_1 = pow(np.real(stat_vector[0]),2)+pow(np.real(stat_vector[1]),2)+pow(np.real(stat_vector[2]),2)+pow(np.real(stat_vector[3]),2)
    total_2 = pow(np.real(stat_vector_2[0]),2)+pow(np.real(stat_vector_2[1]),2)+pow(np.real(stat_vector_2[2]),2)+pow(np.real(stat_vector_2[3]),2)

    return (8*total_1+8*total_2-8)



