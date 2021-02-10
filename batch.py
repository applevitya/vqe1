#!/usr/bin/python3

import subprocess


#for m in range(5, 15, 5):
#    subprocess.run(['./experiment.py', '-ldata/2020_10_24_schwinger_m{:.1f}_AM10_1Ch_2.dat'.format(m), '-m {:.1f}'.format(m), '-N 150'])

def run(m, n=0):
    subprocess.run(['./simulations.py', '-ldata/sim_lcwp_after_SPDC_with_fiber/2020_11_11_schwinger_sim_m{:.1f}_n{:.1f}_2ch.dat'.format(m, n), '-m {:.1f}'.format(m), '-N 150', '-n {:.1f}'.format(n)])

for m in range(-10, 11, 1):
    n = 0.0
    while n <= 1.0:
        run(m, n)
        n += 0.1

