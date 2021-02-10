from math import pi
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


with open('/Users/viktorkrasnikov/Desktop/Vqe_data/vqe:BFGS.txt', 'r') as file:
    nums = [row.strip() for row in file]
nums = list(map(float, nums))


plt.hist(nums,bins=int(len(nums)/100), color = 'blue',density=True)
# Plot formatting

plt.ylabel('')
plt.xlabel('значение энергии')
plt.title('Плотность основного состояния гамильтониана Швингера (m = 0)')
plt.show()
