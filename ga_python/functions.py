import time
from numpy import cos, sqrt, sin, array, pi
from math import pi as π, exp


def Levy3(x):
    time.sleep(0.01)
    x = array(x)
    n = len(x)
    y = 1 + (x - 1) / 4
    # calculate f(y(x))
    term1 = sin(pi*y[0])**2
    term3 = (y[n-1]-1)**2 *(1 + sin(2*pi*y[n-1]))**2

    sum = 0 
    for x_i in y:
        new = (x_i-1)**2 * (1+10*sin(pi*x_i+1)**2)
        sum += new
    return term1+term3+sum


def michalewicz(x, m=10):
    x = array(x)
    return -sum(sin(v)*sin(i*v**2/pi)**(2*m) for (i,v) in enumerate(x))


def ackley(x, a=20, b=0.2, c=2*π):
    x = array(x)
    time.sleep(0.01)
    d = len(x)

    return -a*exp(-b*sqrt(sum(x**2)/d)) - exp(sum(cos(c*xi) for xi in x)/d) + a + exp(1)
