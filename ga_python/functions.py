import time
from numpy import cos, sqrt
from math import pi as π, exp


def ackley(x, a=20, b=0.2, c=2*π):
    time.sleep(0.01)
    d = len(x)
    
    return -a*exp(-b*sqrt(sum(x**2)/d)) - exp(sum(cos(c*xi) for xi in x)/d) + a + exp(1)

