import multiprocessing
from subprocess import call

betas = range(15,45,2)

def run_a_slice(betas):
	for beta in betas:
		call(['python','produce_overall_player_quality.py',str(beta)])
	return

def partition (lst, n):
    return [lst[i::n] for i in xrange(n)]

num_cores = 15
slices = partition(betas,num_cores)
pool = multiprocessing.Pool(num_cores)
results = pool.map(run_a_slice, slices)
pool.close()