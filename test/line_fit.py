from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def line(x,a,b):
    return a*x + b

x = [1,2,3,4,5]
y = [1.1, 2.1, 2.8, 4.3, 5.1]

popt, pcov = optimize.curve_fit(line, x, y)

print popt
print pcov

plt.plot(x,y,"ro")
plt.plot(x,(popt[0]*x+popt[1]),"g--")
plt.grid()
plt.show()
