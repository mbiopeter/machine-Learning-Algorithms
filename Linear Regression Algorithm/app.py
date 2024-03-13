import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('LinearRegression.csv')


def Loss_function(m, c, points):
    total_errors = 0
    for i in range(len(points)):
        x = points.iloc[i].Time
        y = points.iloc[i].Score
        total_errors += (y - (m * x + c)) ** 2
    total_errors / len(float(points))


def gradient_descent(m_now, c_now, points, L):
    m_gradient = 0 
    c_gradient = 0
    n = len(points)
    for  i in range(n):
        x = points.iloc[i].Time
        y = points.iloc[i].Score
        m_gradient += -(2/n) * x * (y - (m_now * x + c_now))      
        c_gradient += -(2/n) * (y - (m_now * x + c_now))      
    m = m_now - m_gradient * L
    c = c_now - c_gradient * L
    return m, c

m = 0
c = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, c = gradient_descent(m, c, data, L)

print(m, c)
plt.scatter(data.Time, data.Score, color='black')
plt.plot(list(range(1, 300)), [m * x + c for x in range(1, 300)], color='red')
plt.show()