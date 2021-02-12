from math import pi
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


with open('Vqe_data/vqe_SLSQP_1grad.txt', 'r') as file:
    nums = [row.strip() for row in file]
nums = list(map(float, nums))


plt.hist(nums,bins=int(len(nums)/20), color = 'blue')
# Plot formatting
plt.xlim(-1.7,-1.3)
plt.ylabel('')
plt.xlabel('значение энергии')
plt.title('Плотность основного состояния гамильтониана Швингера (m = 0)')
plt.show()
