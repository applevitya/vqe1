from qutip import *
from numpy.random import random, multinomial
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import *
from scipy.optimize import *
from matplotlib import pyplot as plt
from tkinter import *

# Create zero and one states
state_zero = np.array([[1.0],
                       [0.0]])
state_one = np.array([[0.0],
                      [1.0]])
def getQobj(psi):
    return Qobj(psi)


# Create Hamiltonian №1
Ht=tensor(sigmaz(),sigmaz())


#Initial state
psi0=state_zero;
def matrevol(x,y):
    D=np.array([[cos(x)+complex(0,sin(x)*cos(2*y)),complex(0,sin(x)*sin(2*y))],[complex(0,sin(x)*sin(2*y)),cos(x)-complex(0,sin(x)*cos(2*y))]])
    return D
def newstate(delta,phi,psi):
    return np.dot(matrevol(delta,phi),psi)
def getquqvarq(psi1):
    psi2=psi1[0][0]*tensor(getQobj(state_zero),getQobj(state_one))+psi1[1][0]*tensor(getQobj(state_one),getQobj(state_zero))
    return psi2
psi1=newstate(pi/4,pi/4,psi0)



#Simulation

#state after Plate
def returnstate(theta2,theta3,theta4,theta5,psi):
    delta2=pi/4
    delta3=pi/2
    delta4=pi/4
    delta5=pi/2
    U1=np.dot(matrevol(delta2,theta2),matrevol(delta3,theta3));
    U2=np.dot(matrevol(delta4,theta4),matrevol(delta5,theta5));
    state=tensor(getQobj(U1),getQobj(U2));
    state=np.dot(state,psi);
    return state
#probabilities of coincidence
def probability(psi,N):
    if N==1:
        M=np.dot((tensor(getQobj(state_zero),getQobj(state_zero))),(np.transpose(tensor(getQobj(state_zero),getQobj(state_zero)))));
        p=np.trace(np.dot(M,np.dot(psi,np.transpose(psi.conj()))))
    if N==2:
        M = np.dot((tensor(getQobj(state_zero), getQobj(state_one))),
                   (np.transpose(tensor(getQobj(state_zero), getQobj(state_one)))));
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    if N==3:
        M = np.dot((tensor(getQobj(state_one), getQobj(state_zero))),
                   (np.transpose(tensor(getQobj(state_one), getQobj(state_zero)))));
        p = np.trace(np.dot(M, np.dot(psi,np.transpose(psi.conj()))))
    if N==4:
        M = np.dot((tensor(getQobj(state_one), getQobj(state_one))),
                   (np.transpose(tensor(getQobj(state_one), getQobj(state_one)))));
        p = np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj()))))
    return p.real

#get value of Hamiltonian
def valueH1(state):
    H=probability(state,1)-probability(state,2)-probability(state,3)+probability(state,4)
    return np.real(H)

a=returnstate(pi/4,pi/4,pi/4,pi/4,getquqvarq(psi1));
#print(valueH1(a))



#Simulation
     # range N from 0 to 3: probabilities for HH,HV,VH,VV

def probabilityRandom(a,N,n):
    probabilityArray=np.random.multinomial(n, [probability(a,1),probability(a,2),probability(a,3),probability(a,4)], 1)
    return probabilityArray[0][N]
def simulation(*args):
     n=args[0][4]
     state=returnstate(args[0][0],args[0][1],args[0][2],args[0][3],getquqvarq(psi1))
     H=probabilityRandom(state,0,n)-probabilityRandom(state,1,n)-probabilityRandom(state,2,n)+probabilityRandom(state,3,n)
     return H/n


def arr(*args):
   print(args[0][0])
def callbackxk(xk):
    xklist.append(xk)
    #print(xklist)
    return False
def error(xk):
    return abs(-1-simulation(xk))

#текущая точность
current=[]
nstat=[]
for k in range(0,2):
    nn=2
    ylist=[]
    x0=[1,2,3,2,(1000**k+100)]

    for i in range(0,nn):
        xklist=[]
        xlist=range(0,len(xklist))
        minimize(simulation, x0, method="Nelder-Mead", callback=callbackxk, options={'maxiter': 100})
        ylist.append([log(10e-10+abs(1+simulation(x)),10) for x in xklist])
    xlist = range(0, len(xklist))

    #plt.plot (xlist, ylist[0])


    sumlist=[]
    for j in range(0,len(xklist)):
        averg=0
        for i in range(0,nn):
            averg+=ylist[i][j]
        averg=averg/(nn)
        sumlist.append(averg)
    #plt.plot(xlist,sumlist)
    current.append(min(sumlist))
    nstat.append(np.log10(x0[4]))

plot1=plt.plot(nstat,current)

plot1=plt.show()




