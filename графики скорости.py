import numpy as np
import math as m
import matplotlib.pyplot as plt
from icecream import ic
ic.enable()

d = 0.4
mass = 180

g = 8.87
S = (m.pi * d ** 2)/4
tetta = -9 #* (m.pi / 180)
V = np.float64(11_000)  # Используем тип данных float64
Cx = 0.9
M = []
square = []
speed = []

while mass > 0:
    V = m.sqrt((2 * mass * g) / (Cx * 64.79 * S))
    speed.append(V)
    M.append(mass)
    ic(mass, V)
    mass -= 0.5

   
plt.plot(M, speed)
plt.title('Зависимость скорости от массы')
plt.xlabel('Масса, кг')
plt.ylabel('Скорость, м/с')
plt.grid(True)
plt.show()
'''
while S >= 0.12:
    V = m.sqrt((2 * mass * g) / (0.25 * 64.79 * S))
    speed.append(V)
    square.append(S)
    S -= 0.01

plt.plot(square, speed)
plt.title('Зависимость скорости от площади миделево сечения')
plt.xlabel('Миделево сечение, м^2')
plt.ylabel('Скорость, м/с')
plt.grid(True)
plt.show()
'''