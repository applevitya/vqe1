from math import pi
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


with open('Vqe_data/vqe_bad_grad.txt', 'r') as file:
    nums = [row.strip() for row in file]
nums = list(map(float, nums))


plt.hist(nums,bins=int(len(nums)/40), color = 'blue')
# Plot formatting
plt.xlim(-1.6,-1.25)
plt.ylabel('')
plt.xlabel('значение энергии')
plt.title('Плотность основного состояния гамильтониана Швингера (m = 0)')
plt.show()
