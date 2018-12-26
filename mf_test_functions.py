import numpy as np
import math

def mfShekel(args, fidelity):
    x = np.array(args)
    numFidelities = 3
    assert 0 <= fidelity < numFidelities
    # Defines the locations of the local extrema.
    # A.shape = (m, dim) for m local extrema.
    a = np.array([[4., 4., 4., 4.],
                [1., 1., 1., 1.],
                [8., 8., 8., 8.],
                [6., 6., 6., 6.],
                [3., 7., 3., 7.],
                [2., 9., 2., 9.],
                [5., 5., 3., 3.]])
    deltaA = np.array([[1., -1., 1., 1.],
                    [-1., 1., 1., 1.],
                    [1., 1., -1., -1.],
                    [1., 1., 1., 1.],
                    [1., -1., 1., -1.],
                    [-1., -1., 1., 1.],
                    [-1., -1., -1., -1.]])
    # Defines the magnitude of the local extrema.
    c = np.array([.1, .2, .2, .4, .4, .6, .3])
    deltaC = np.array([.05, -.05, .1, .05, -.05, -.1, .05])
    assert a.shape[1] == x.shape[0]

    def shekel(x, a, c):
        '''
        Shekel function
        (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)
        0 <= x <= 10
        Global maximum: f(4, 4, 4, 4) = 10.402818836930305
        '''
        return np.sum(1. / (np.sum((x - a) ** 2., axis=1) + c))

    epsilon = 0.1 * (numFidelities - 1 - fidelity)
    cost = 2. ** (fidelity - numFidelities + 1)
    return shekel(x, a + epsilon * deltaA, c + epsilon * deltaC), cost

def mfUnivariate(args, f):
    x, = args
    numFidelities = 2
    assert 0 <= f < numFidelities

    def univariate(x):
        '''
        0 <= x <= 1.2
        Global maximum: f(0.96609) = 1.48907
        '''
        return (1.4 - 3. * x) * math.sin(18. * x)

    def error(x, f):
        return (numFidelities - 1 - f) * .3 * math.sin(31. * x - 1)

    return univariate(x) + error(x, f), 10 ** (f - numFidelities + 1)

def mfRosenbrock(args, f):
    x = np.array(args)
    numFidelities = 3
    assert(0 <= f and f < numFidelities)

    def rosenbrock(x):
        '''
        Rosenbrock Function
        (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)
        -2 <= xi <= 2
        The global minima is located at x = f(1, ..., 1), f(x) = 0.
        '''
        return np.sum(100. * (x[1:] - x[:-1] ** 2.) ** 2. + (x[:-1] - 1) ** 2.)

    def error(x, f):
        offsets = [3.5, 1.1, 0.]
        epsilon = 250.
        resolution = 1.5
        x1, x2 = x
        r = (math.sin(resolution * x1 + offsets[f])
           * math.cos(resolution * x2 + math.sin(offsets[f])) ** 2.)
        return epsilon * (numFidelities - 1 - f) * r

    cost = 10. ** (f - numFidelities + 1)
    return rosenbrock(x) + error(x, f), cost

def mfHosaki(args, f):
    x = np.array(args)
    numFidelities = 3
    assert(0 <= f and f < numFidelities)

    def hosaki(x):
        '''
        Hosaki Function
        (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)
        Domain: [0, 10]^2
        The global minimum is located at x = f(4, 2), f(x) = -2.345811576101292
        '''
        x1, x2 = x
        p = 1. - 8. * x1 + 7. * x1 ** 2. - 7./3. * x1 ** 3. + .25 * x1 ** 4.
        return p * x2 ** 2. * math.exp(-x2)

    def error(x, f):
        offsets = [4.1, 3.2, 0.]
        epsilon = 0.5 * (numFidelities - 1 - f)
        resolution = 1.
        x1, x2 = x
        r = (math.sin(resolution * x1 + offsets[f])
           * math.cos(resolution * x2 + math.sin(offsets[f])) ** 2.)
        return epsilon * r

    cost = 10. ** (f - numFidelities + 1)
    return hosaki(x) + error(x, f), cost

def mfHartmann3(x, f):
    if f == 2:
        return cfHartmann3(x, np.array([1,1,1,1])), cost_hartman(np.array([1,1,1,1]))
    elif f == 1:
        return cfHartmann3(x, np.array([.667,.667,.667,.667])), cost_hartman(np.array([.667,.667,.667,.667]))
    elif f == 0:
        return cfHartmann3(x, np.array([.331,.331,.331,.331])), cost_hartman(np.array([.331,.331,.331,.331]))


def cost_hartman(z):
    return .005 + (1-.05)*pow(z[0],3)*pow(z[1],2)*pow(z[2],1.5)*z[3]

def cfHartmann3(x, z):
    alpha = np.array([1., 1.2, 3., 3.2])
    return hartmann3(x, alpha-0.1*alpha*(1-z))

def mfHartmann6(x, f):
    numFidelities = 4
    assert 0 <= f and f < numFidelities
    alpha = np.array([1., 1.2, 3., 3.2])
    delta = np.array([0.001, -0.001, -0.01, 0.01])
    cost = 10. ** (f - numFidelities + 1)
    return hartmann6(x, alpha + delta * (numFidelities - f - 1)), cost

def hartmann(x, A, P, alpha):
    result = 0.
    for i in range(4):
        tmp = 0.
        for j in range(len(x)):
            tmp -= A[i][j] * (x[j] - P[i][j]) ** 2.;
        result -= alpha[i] * math.exp(tmp)
    return result
'''
http://www.sfu.ca/~ssurjano/hart3.html
Domain: (0, 1)^3
Gloabal Minimum: f(x) = -3.86278
    for x = (0.114614, 0.555649, 0.852547)
'''
def hartmann3(x, alpha):

    A = [[3., 10., 30.],
        [0.1, 10., 35.],
        [3., 10., 30.],
        [0.1, 10., 35.]]

    P = [[.3689, .1170, .2673],
        [.4699, .4387, .7470],
        [.1091, .8732, .5547],
        [.0381, .5743, .8828]]

    return hartmann(x, A, P, alpha)

'''
http://www.sfu.ca/~ssurjano/hart6.html
The 6-dimensional Hartmann function has 6 local minima.
Domain: (0, 1)^6
Global Minimum: f(x) = -3.32237 (Made negative to be a maximum)
    for x = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
'''
def hartmann6(x, alpha):

    A = [[10., 3., 17., 3.5, 1.7, 8.],
        [0.05, 10., 17., 0.1, 8., 14.],
        [3., 3.5, 1.7, 10., 17., 8.],
        [17., 8., 0.05, 10., 0.1, 14.]]

    P = [[.1312, .1696, .5569, .0124, .8283, .5886],
        [.2329, .4135, .8307, .3736, .1004, .9991],
        [.2348, .1451, .3522, .2883, .3047, .6650],
        [.4047, .8828, .8732, .5743, .1091, .0381]]

    return hartmann(x, A, P, alpha)

def mfPark1(x, f):
    if f == 1:
        return park1(x), 1.
    elif f == 0:
        return lowFidelityPark1(x), 0.1

'''
https://www.sfu.ca/~ssurjano/park91a.html
According to Mathematica:
Global Max: f(x) = 25.5893 where x = (1, 1, 1, 1) on [0, 1]^4.
'''
def park1(x):
    x1, x2, x3, x4 = x

    a = math.sqrt(1. + (x2 + x3**2.) * x4 / x1**2.) - 1.
    b = (x1 + 3.*x4) * math.exp(1. + math.sin(x3))
    return 0.5*x1 * a + b

def lowFidelityPark1(x):
    x1, x2, x3, _ = x

    p = -2.*x1 + x2**2. + x3**2. + 0.5
    return (1. + math.sin(x1)/10.) * park1(x) + p

def mfPark2(x, f):
    if f == 1:
        return park2(x), 1.
    elif f == 0:
        return lowFidelityPark2(x), 0.1

'''
https://www.sfu.ca/~ssurjano/park91b.html
According to Mathematica:
Global Max: f(x) = 5.92604 where x = (1, 1, 1, 0) on [0, 1]^4.
'''
def park2(x):
    x1, x2, x3, x4 = x

    return 2./3. * math.exp(x1 + x2) - x4 * math.sin(x3) + x3

def lowFidelityPark2(x):
    return 1.2 * park2(x) - 1.

def mfCurrinExp(x, f):

    return cMFCurrinExp(x,f),cost_currin(f)



def cMFCurrinExp(args, z):
    x, y = args

    p = 2300. * x ** 3. + 1900. * x ** 2. + 2092. * x + 60.
    q = 100. * x ** 3. + 500. * x ** 2. + 4. * x + 20
    if y == 0:
        r = 0
    else:
        r = math.exp(-.5 / y)

    return (1 - 0.1*(1-z)*r) * (p / q)


def cost_currin(z):
    return .1+pow(z,2)

def mfBadCurrinExp(x, f):
    if f == 1:
        return currinExponential(x), 1.
    elif f == 0:
        return -currinExponential(x), 0.1

'''
https://www.sfu.ca/~ssurjano/curretal88exp.html
According to Mathematica:
Global Max: f(x) = 13.7987 where x = (0.216667, 0.0228407) on [0, 1]^2.
'''
def currinExponential(args):
    x, y = args

    p = 2300. * x ** 3. + 1900. * x ** 2. + 2092. * x + 60.
    q = 100. * x ** 3. + 500. * x ** 2. + 4. * x + 20
    if y == 0:
        r = 0
    else:
        r = math.exp(-.5 / y)

    return (1. - r) * p / q

def lowFideliltyCurrinExponential(args):
    x, y = args
    a = (x + .05, y + .05)
    b = (x + .05, max(0, y - .05))
    c = (x - .05, y + .05)
    d = (x - .05, max(0, y - .05))

    return .25 * (currinExponential(a) + currinExponential(b)
                + currinExponential(c) + currinExponential(d))

def mfBorehole(x, f):

    return cfBorehole(x,f),cost_borehole(f)



def cfBorehole(args,z):
    return z*borehole(args) + (1-z)*lowFidelityBorehole(args)


def cost_borehole(z):
    return .1+pow(z,1.5)

'''
https://www.sfu.ca/~ssurjano/borehole.html
Global max: 309.523221
Bounds: [0.05 0.15; ...
         100, 50000; ...
         63070, 115600; ...
         990, 1110; ...
         63.1, 116; ...
         700, 820; ...
         1120, 1680; ...
         9855, 12045];
'''
def borehole(x):
    (rw, r, Tu, Hu, T1, H1, L, Kw) = x
    frac1 = 2. * math.pi * Tu * (Hu - H1)
    frac2a = 2. * L * Tu / (math.log(r / rw) * rw ** 2. * Kw)
    frac2b = Tu / T1
    frac2 = math.log(r / rw) * (1. + frac2a + frac2b)
    return frac1 / frac2

def lowFidelityBorehole(x):
    (rw, r, Tu, Hu, T1, H1, L, Kw) = x
    frac1 = 5. * Tu * (Hu - H1)
    frac2a = 2. * L * Tu / (math.log(r / rw) * rw ** 2. * Kw)
    frac2b = Tu / T1
    frac2 = math.log(r / rw) * (1.5 + frac2a + frac2b);
    return frac1 / frac2;


def mfReal(x, f, mfobject):
    import time
    t1 = time.time()
    value = mfobject.eval_at_fidel_single_point_normalised([f, 1.0], x)
    t2 = time.time()
    t0 = t2 - t1
    return value, t0