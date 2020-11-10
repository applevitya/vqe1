from qiskit import *
from qiskit.visualization import plot_histogram

from qiskit.tools.visualization import circuit_drawer
from math import *



qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])


backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend,shots=100)
plt = plot_histogram(job.result().get_counts(), color='midnightblue', title="New Histogram")

qc.draw('mpl').show()


plt.show()