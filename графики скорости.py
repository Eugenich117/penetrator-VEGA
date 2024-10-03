import numpy as np
import math as m
import matplotlib.pyplot as plt
from icecream import ic
ic.enable()

d = 4
mass = 180

g = 8.87
S = (m.pi * d ** 2)/4
tetta = -9 #* (m.pi / 180)
V = np.float64(11_000)  # Используем тип данных float64
Cx = 0.9
M = []
square = []
speed = []
'''
while mass > 0:
    V = m.sqrt((2 * mass * g) / (Cx * 64.79 * S))
    speed.append(V)
    M.append(mass)
    ic(mass, V)
    mass -= 0.5

   
plt.plot(M, speed)
plt.title('Зависимость скорости от массы', fontsize=16, fontname='Times New Roman')
plt.xlabel('Масса, кг', fontsize=16, fontname='Times New Roman')
plt.ylabel(r'Скорость, $\frac{м}{с}$', fontsize=16, fontname='Times New Roman')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.grid(True)
plt.show()
'''
while S >= 0.12:
    V = m.sqrt((2 * mass * g) / (0.25 * 64.79 * S))
    speed.append(V)
    square.append(S)
    S -= 0.01

plt.plot(square, speed)
plt.title('Зависимость скорости от площади миделево сечения', fontsize=16, fontname='Times New Roman')
plt.xlabel('Миделево сечение, м$^2$', fontsize=16, fontname='Times New Roman')
plt.ylabel(r'Скорость, $\frac{м}{с}$', fontsize=16, fontname='Times New Roman')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.grid(True)
plt.show()
