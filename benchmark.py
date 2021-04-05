import math
import random
import numpy as np 
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from random import random, uniform, gauss
from math import exp, log, sqrt, ceil


def Sphere(x):
    return sum(x**2)

def Elliptic(x):
    sm = 0
    n = len(x)
    for i in range(n):
        a = (10**6)**(i/(n-1))
        sm += a * (x[i])**2
    return sm      
    

def SumSquare(x):
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**2 )

def SumPower(x):
    sm = 0
    for i in range(len(x)):
        sm += abs(x)**(i+2)
    return sm        

def Schwefel222(x):    
    return sum(abs(x))+prod(abs(x))

def Schwefel221(x):
    return abs(x).max()

def Step(x):
    n = len(x)
    sm = 0
    for i in range(len(x)):
        sm += (np.floor(x) + 0.5 )**2
    return sm

def QuarticWn(x):
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**4 ) + np.random.random()

def Quartic(x):
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**4 )

def Rosenbrock(x):
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )+ 100 * sum( (x1 - x0**2) **2 ))

def Rastrigin(x):
    n = len(x)
    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))

def Rastrigin_noncont(x):
    n = len(x)
    for i in range(n):
        if abs(x[i]) >= 0.5:
            np.round(2*x[i])/2          
        
    return sum(x**2 - (10*cos(2*pi*x)+10))

def Griewank(x):
    n = len(x)
    s = sum( x**2 )
    p = 1
    for i in range(n):
        p *= cos(x[i]/sqrt(i+1))
    return s/4000 - p + 1

def Schwefel226(x):
    n = len(x)
    sm = 0
    for i in range(n):
        sm += x[i]*sin(sqrt(abs(x[i])))
    return 418.9829*n - sm

def Ackley(x):
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( 2*pi * x ))
    return -20 *exp( -0.2*sqrt( s1 / n )) - exp( s2 / n ) + 20 + exp(1)

def Penalized1(x):
    n = len(x)
    p = x
    p_ = 1+ 1/4*(x+1)
    p1 = p_[:n-1]
    p2 = p_[1:]
    p3 = p
    for i in range(n):
        if p[i] > 10:
            p3[i] = 100*(p[i]-10)**4
        elif p[i] <-10 :
            p3[i] = 100*(-p[i]-10)**4
        elif (p[i] >= -10) & (p[i] <= 10):
            p3[i] = 0
    
    p = pi/n * ( 10*sin(pi*p_[0])**2 + sum( ((p1-1)**2)*(1+10*sin(pi*p2)**2) ) + (p_[n-1]-1)**2 ) + sum(p3)    
    return p

def Penalized2(x):
    n = len(x)
    p = x
    p1 = p[:n-1]
    p2 = p[1:]
    p3 = p
    for i in range(n):
        if p[i] > 5:
            p3[i] = 100*(p[i]-5)**4
        elif p[i] <-5 :
            p3[i] = 100*(-p[i]-5)**4
        elif (p[i] >= -5) & (p[i] <= 5):
            p3[i] = 0
    p = 0.1 * ( (sin(pi*p[0])**2) + sum( ((p1-1)**2)*(1+sin(3*pi*p2)**2) ) + ((p[n-1]-1)**2)*(1+sin(2*pi*p[n-1])**2) ) + sum(p3)
    return 0

def Alpine(x):    
    return sum (abs( x*sin(x) + (0.1*x) ))

def Levy(x):
    n = len(x)
    z = 1 + (x - 1) / 4
    return (sin( pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))

def Weierstrass(x):
    a = 0.5
    b = 3
    k_max = 20

    def sub_sum(x):
        return sum([a**k * np.cos(2*math.pi*(b**k)*(x + 0.5)) for k in range(k_max)])

    val = sum([sub_sum(x0) for x0 in x]) - (len(x) * sum([a**k * np.cos(2*math.pi*(b**k)*0.5) for k in range(k_max)]))

    return val


def Schaffer(x):
    pay = (sin( sum(x**2)**0.5 )**2) - 0.5
    payda = (1+0.001*sum(x**2))**2
    
    return (0.5 + (pay/payda))



"""
benchmark fonksiyonları

Sphere [-100, 100]
Elliptic [-100, 100]
SumSquare [-10, 10]
SumPower [-10, 10]
Schwefel222 [-10, 10]
Schwefel221 [-100, 100]
Step  [-100, 100]
QuarticWn [-1.28, 1.28]
Quartic [-1.28, 1.28]
Rosenbrock [-10, 10]
Rastrigin [-5.12, 5.12]
Rastrigin_noncont [-5.12, 5.12]
Griewank [-600, 600]
Schwefel226 [-500, 500]
Ackley [-32, 32]
Penalized1 [-50, 50]
Penalized2 [-50, 50]
Alpine [-10, 10]
Levy [-10, 10]
Weierstrass [-0.5, 0.5]
Schaffer [-100, 100]


"""
