from joblib import Parallel, delayed
import multiprocessing
from simulations import optimization

n_cores = multiprocessing.cpu_count()

r = Parallel(n_jobs=n_cores)(delayed(optimization)() for i in range(4));

with open("hello.txt", "w") as file:
    for line in r:
        file.write(str(line) + '\n')


