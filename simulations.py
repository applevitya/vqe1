#!/usr/bin/python3

import logging
from optparse import OptionParser
from sys import stdout
import random

import numpy as np
from math import pi
from matplotlib import pyplot as plt
from vqe import *
from gradients import *
import datetime
from scipy.optimize import minimize
from examples import energy
from examples import energy,schwinger_samples

# Log functions
def log_header(setup, H, logfile):
    logfile.write('# VQE log started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    logfile.write('# Hamiltonian = ' + str(H.hamiltonian) + '\n')
    logfile.write('# Exposure time = {:d} ms'.format(setup.exposure_time))
    if (setup.fake):
        logfile.write(', simulations with intensity = {:.1f} Hz, noise = {:.3f}\n'
                      .format(setup.intensity, setup.noise))
    else:
        logfile.write('\n')
    logfile.write('# Efficiencies = ' + str(setup.efficiencies) + '\n')
    logfile.write('#')
    for i in range(6):
        logfile.write('{:>7s} '.format('x' + str(i)))
    logfile.write('{:>7s} '.format('Hmean'))
    logfile.write('{:>7s}\n'.format('Hstdev'))
    logfile.flush()


def log_data(logfile, x, res):
    logfile.write(' ')
    for xi in x:
        logfile.write('{:>7.3f} '.format(xi))
    logfile.write('{:>7.3f} '.format(res[0]))
    logfile.write('{:>7.3f}\n'.format(res[1]))
    logfile.flush()


# Option parser
parser = OptionParser()
parser.add_option('-f', '--fake', action='store_true', default=True,
                  help='perform simulations instead of real experiment. Always on.')
parser.add_option('-l', '--log', default='vqe.dat',
                  metavar='FILE', help='write output to FILE. Default "%default"')
parser.add_option('-t', '--exposure-time', type='int', default=5000,
                  metavar='TIME', help='set exposure time to TIME milliseconds. Default %default')
parser.add_option('-N', '--iterations', type='int', default=550,
                  metavar='ITER', help='number of iterations for optimization algorithm. Default %default')
parser.add_option('-i', '--intensity', type='float', default=4000,
                  metavar='HERTZ', help='intensity in Hertz that is used for simulations. Default %default')
parser.add_option('-v', '--verbosity', default='warn',
                  choices=['debug', 'info', 'warn', 'error'],
                  metavar='LEVEL',
                  help='set verbosity level of additional messages to LEVEL. Available: debug, info, warn, error. Default %default')
parser.add_option('-m', type='float', default=0,
                  metavar='VAL', help='field parameter in Schwinger Hamiltomian. Default %default')
parser.add_option('-n', '--noise', type='float', default=0,
                  metavar='STRENGTH', help='specify decoherence noise strength for simulations. Default %default')
(options, args) = parser.parse_args()

options.verbosity = verbosity = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR
}[options.verbosity]

# Hamiltonians
ZZ = {'hh': (-1, 1, 1, -1)}  # -ZZ
ZZXX = {'hh': (-1, 1, 1, -1), 'dd': (-2, 0, 0, 2)}  # -ZZ-XI-IX
ZZYY = {'hh': (-1, 1, 1, -1), 'rr': (-2, 0, 0, 2)}  # -ZZ-YI-IY


def schwinger(m):
    return {'hh': (1, -m, 1 + m, 2), 'dd': (1, -1, -1, 1), 'rr': (1, -1, -1, 1)}  # 1+XX+YY+0.5(-ZI+ZZ+mIZ-mZI)


# 'hh':(1,-m,1+m,2)

# Initialization
logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=options.verbosity)
setup = Setup(fake=options.fake)
setup.set_exposure_time(options.exposure_time)
setup.intensity = options.intensity  # Intensity in Hertz for simulations
setup.noise = options.noise
setup.efficiencies = [0.955631, 0.954148, 0.975911, 1.10216]
setup.retardances = [pi / 2, 2 * pi * 0.4995, 2 * pi * 0.2493, 2 * pi * 0.5129, 2 * pi * 0.2198,
                     2 * pi * 0.4799]  # Waveplace retardances
# Hamiltonian evaluator
H = MeanValue(setup, schwinger(options.m))


# logfile = open(options.log, 'w')
# log_header(setup, H, stdout)
# log_header(setup, H, logfile)


# Optimization

x0 = np.random.uniform(0, 2 * pi, 6)
def optimization():
    points = []
    def callback_func(x):
        points.append(schwinger_samples(x,100000))
        return False
        # log_data(stdout, x, result)
        # log_data(logfile, x, result)



    def target_func(x):
        return schwinger_samples(x,100000) #np.real(energy(x))



    # x0 = [5.04116483e-03, 1.09884935e+00, 8.26706055e-01, 3.41551829e+00,4.01166258e+00, 5.30849340e+00]
    # x0 = [-8.96511828, 4.62359229, 4.44776618, 3.25604477, 1.46245918, 1.77980848]
    # Gradient for SLSQP#########################
    def gradient_slsqp(x0):
        r = 3 / 2
        der = np.zeros_like(x0)
        x = np.copy(x0)
        for i in range(len(x0)):
            x[i] = x0[i] + pi / (4 * r)
            der[i] = r * target_func(x)
            x[i] = x0[i] - pi / (4 * r)
            der[i] -= r * target_func(x)
            x[i] = x0[i]
        return der

    def gradient_slsqp_2(x0):
        der = np.zeros_like(x0)
        for i in range(0, len(x0)):
            j = i + 1
            der[i] = hadamard_test(x0, j)
        return der

    m = options.m
    #result = minimize_spsa(target_func, callback=callback_func, x0=x0, maxiter=options.iterations,a0=0.01, af=0.01, b0=0.1, bf=0.02)
    # a0=0.05/(0.2*abs(m)+1), af=0.005/(0.2*abs(m)+1), b0=0.1, bf=0.02)
    result = minimize(target_func, x0=x0, callback=callback_func, method="SLSQP",jac = gradient_slsqp_2,options={'disp':True, 'maxiter': 110, 'eps': 0, "ftol":0})
    iteration_number = [i for i in range(0, len(points))]
    plt.scatter(iteration_number, points, color='g', linestyle='--')

    # Order operator
    O = MeanValue(setup, {'hh': (0, 0, 1, 0)})
    # log_data(stdout, result.x, O(result.x))
    # log_data(logfile, result.x, O(result.x))
    # min(min(xx))

    return target_func(result.x)


d1 = []
for j in range(0, 1, 1):
    d2 = optimization()
    d1.append(d2)
    print(d2)

#plt.hist(d1, bins=10)
plt.show()

