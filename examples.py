import qiskit as q
from math import *


qc = q.QuantumCircuit(1)
qc.x(0)
qc.rz(pi/4, 0)






backend = q.Aer.get_backend('statevector_simulator') # Tell Qiskit how to simulate our circuit
result = q.execute(qc,backend).result()
out_state = result.get_statevector()
print(out_state)


