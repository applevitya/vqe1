from math import pi
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


with open('Vqe_data/vqe_hadamard_slsqp.txt', 'r') as file:
    nums = [row.strip() for row in file]
nums = list(map(float, nums))


plt.hist(nums,bins=int(len(nums)/100), color = 'blue')
# Plot formatting
#plt.xlim(-1.6,-1.25)
plt.ylabel('')
plt.xlabel('значение энергии')
plt.title('Плотность основного состояния гамильтониана Швингера (m = 0) Hadamard test')
plt.show()
