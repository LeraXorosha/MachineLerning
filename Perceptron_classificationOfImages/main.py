import numpy as np
import matplotlib.pyplot as plt

N = 5 # по 5 образов для одного и другого класса
b = 3 # смещение прямой по оси y
w1 = -0.5
w2 = 0.5
w3 = -b*w2

x1 = np.random.random(N) #случайные величины по оси x1
x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b #x2 моделируется, как х1 + случайное отклонений
C1 = [x1,x2] #получились точки лежащие выше прямой

x1 = np.random.random(N) #случайные величины по оси x1
x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 + b #x2 моделируется, как х1 - случайное отклонений, где (-0.1), гарантирует что значения будут ниже прямой
C2 = [x1,x2] #получились точки лежащие ниже прямой

f = [0+b,1+b] #разделяющая прямая

w = np.array([w1, w2, w3]) #веса
for i in range(N):
    x = np.array([C1[0][i], C1[1][i], 1]) #перебор всех образов для класса С2
    y = np.dot(w,x) # вычисление выходного значение
    if y>=0:
        print("Класс С1")
    else:
        print("Класс С2")

#отображение точек
plt.scatter(C1[0][:], C1[1][:], s=10,c='red')
plt.scatter(C2[0][:], C2[1][:], s=10,c='blue')
plt.plot(f) #отрисуем линию
plt.grid(True)
plt.show()
