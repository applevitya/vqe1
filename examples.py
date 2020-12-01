from qiskit import *
from qiskit.visualization import plot_histogram

from qiskit.tools.visualization import circuit_drawer
from math import *


############## Circuit ################
qc = QuantumCircuit(3,1)
qc.h(0)
qc.h(1)
qc.ccx(0,1,2)

#######################################


########### State_vector ##################
simulation = Aer.get_backend('statevector_simulator')
print(execute(qc,simulation).result().get_statevector(qc))
##########################################

# Measuremetns###
qc.measure(2,0)
#qc.measure([0, 1], [0, 1])
backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend)
plt = plot_histogram(job.result().get_counts(qc), color='midnightblue', title="New Histogram")
qc.draw('mpl').show()
plt.show()

