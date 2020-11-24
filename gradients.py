from qiskit import *
from qiskit.visualization import plot_histogram
from vqe import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate, HGate,XGate, IGate, CXGate



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


def derivative_U(phi,delta):
    Y = Operator(RYGate(2 * phi))
    Z = Operator(RZGate(delta))

    return 2*Operator(RYGate(2*phi+pi/2)).compose(Z.conjugate(),front = True).compose(Y.transpose(),front =True)- 2*Y.compose(Z.conjugate(),front = True).compose(Operator(RYGate(-2*phi+pi/2)))


def U_circuit(phi,N):
    if N==0:
        return (((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(
            Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))

    if N == 1:
        return (((derivative_U(phi[0],pi/2).compose(U(phi[1],pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(Operator(CXGate))).compose((U(phi[2],pi/2).compose(U(phi[3],pi))).tensor((U(phi[4],pi/2).compose(U(phi[5],pi)))))
    if N==2:
        return (((U(phi[0], pi / 2).compose(derivative_U(phi[1], pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N==3:
        return (((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (derivative_U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N==4:
        return (((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (U(phi[2], pi / 2).compose(derivative_U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N==5:
        return (((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((derivative_U(phi[4], pi / 2).compose(U(phi[5], pi)))))
    if N==6:
        return (((U(phi[0], pi / 2).compose(U(phi[1], pi))).tensor(Operator(XGate).compose(Operator(IGate)))).compose(
            Operator(CXGate))).compose(
            (U(phi[2], pi / 2).compose(U(phi[3], pi))).tensor((U(phi[4], pi / 2).compose(derivative_U(phi[5], pi)))))





