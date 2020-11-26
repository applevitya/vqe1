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


def derivative_U(phi, delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))

    return 2 * Operator(RYGate(2 * phi + pi / 2)).compose(Z.conjugate(), front=True).compose(Y.transpose(),
                                                                                             front=True) - 2 * Y.compose(
        Z.conjugate(), front=True).compose(Operator(RYGate(-2 * phi + pi / 2)))


def U_circuit(phi, N):
    if N == 0:
        return ((U(1, pi / 2).compose(U(1, pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(
            Operator(CXGate())).compose((U(2, pi / 2).compose(U(2, pi))).tensor((U(2, pi / 2).compose(U(2, pi)))))

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
    phi =[1, 12, 23, 34, 44, 15]
    return (U_circuit(phi, 0).conjugate().transpose()).compose(Operator(HGate()).tensor(Operator(IGate())).compose(U_circuit(phi, N)))


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


    return C

# print((((U(1, pi / 2).compose(U(1, pi))).tensor(Operator(XGate()).compose(Operator(IGate())))).compose(Operator(CXGate())).compose((U(2, pi / 2).compose(U(2, pi))).tensor((U(2, pi / 2).compose(U(2, pi))))).data))

phi = [1, 12, 23, 34, 44, 15]
print(C_Gate(B(1,2),3))

