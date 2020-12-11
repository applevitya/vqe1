from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate


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

def derivative_U_12313131431(phi, delta):
    T =  - 2j * sin(delta / 2) * sin(2 * phi)
    R = 2j * sin(delta / 2) * cos(2 * phi)
    return Operator([[T, R], [-R.conjugate(), T.conjugate()]])

def derivative_U(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))

    return  Operator(RYGate(2 * phi + pi/2)).compose(Z.conjugate(), front=True).compose(Y.transpose(),front=True)

def U_circuit(phi, N):
    if N == 0:
        return ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate())).compose((U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(
            Operator(XGate()).compose(Operator(IGate())))).compose(Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 2:
        return (((U(phi[0], pi / 2).compose(derivative_U(phi[1], pi))).tensor(
            Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 3:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (derivative_U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 4:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(derivative_U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 5:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((derivative_U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 6:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(derivative_U(phi[5], pi)))))


def B(phi, N):

    return (U_circuit(phi, 0).conjugate().transpose()).compose(Operator(IGate()).tensor(Operator(HGate())).compose(U_circuit(phi, N),front= True),front=True)


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
################################################################################################
def derivative_U2(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))
    return -Y.compose(Z.conjugate(), front=True).compose(Operator(RYGate(-2 * phi + pi / 2)), front= True)

def U_circuit2(phi, N):
    if N == 0:
        return ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate())).compose((U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U2(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(
            Operator(XGate()).compose(Operator(IGate())))).compose(Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 2:
        return (((U(phi[0], pi / 2).compose(derivative_U2(phi[1], pi))).tensor(
            Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 3:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (derivative_U2(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 4:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(derivative_U2(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 5:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((derivative_U2(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N == 6:
        return (
        ((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate()))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(derivative_U2(phi[5], pi)))))

def B2(phi, N):

    return (U_circuit2(phi, 0).conjugate().transpose()).compose(Operator(IGate()).tensor(Operator(HGate())).compose(U_circuit2(phi, N),front= True),front= True)







phi = [130, 192, 17, 51, 160, 31]



def hadamard_test(phi,N):
    qc = QuantumCircuit(3, 1)
    qc.h(2)
    qc.append(C_Gate(B(phi,N),3),[0,1,2])
    qc.h(2)
    qc.measure(2, 0)

    qc2 = QuantumCircuit(3, 1)
    qc2.h(2)
    qc2.append(C_Gate(B2(phi, N), 3), [0, 1, 2])
    qc2.h(2)
    qc2.measure(2, 0)




    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=10000)
    job2 = execute(qc2, backend,shots=10000)
    total = job.result().get_counts(qc)['0']
    total = (8*total + 8*job2.result().get_counts(qc2)['0'])/10000 - 8

    #plt = plot_histogram(job.result().get_counts(qc), color='midnightblue', title="New Histogram")
    #qc.draw('mpl').show()
    #plt.show()

    return total
