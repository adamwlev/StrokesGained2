import subprocess
import sys
import os
import itertools
import commands
import multiprocessing

if __name__=="__main__":
	
	# eps = ['300']
	# e_t = ['0.15','0.4']
	# e_d = ['0.23','0.66']
	# w_d = ['0.7','0.95']
	# alpha = ['0.9','0.99']
	# beta = ['5','11','15']

	eps = ['300']
	e_t = ['0.65']
	e_d = ['0.45']
	w_d = ['0.5']
	alpha = ['0.999']
	beta = ['3.5','7','12.5','21','25','30']

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

	def run_a_slice(cats):
		for cat in cats:
			if not os.path.exists('./../ranks/ranks-%s-%s-%s-%s-%s-%s/%s_ranks.npy' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_,cat)): 
				print 'Running produceRanks %s-%s-%s-%s-%s-%s %s' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_,cat)
				subprocess.call(["python","produceRanks.py" ,"%s" % cat,"%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_, "%s" % alpha_, "%s" % beta_])
				cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/ranks/ranks-%s-%s-%s-%s-%s-%s ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/ranks/" % (eps_,e_d_,e_t_,w_d_,alpha_,beta_)
				os.system(cmd)

	def partition (lst, n):
	    return [lst[i::n] for i in xrange(n)]

	for eps_,e_t_,e_d_,w_d_,alpha_,beta_ in itertools.product(*[eps,e_t,e_d,w_d,alpha,beta]):
		if not os.path.exists('./../cats/cats_w%s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)):
			print 'Running SaveShotsBlocks %s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)
			subprocess.call(["python","SaveShotsBlocks.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
		else:
			n_files = len(commands.getstatusoutput('find ./../cats/cats_w%s-%s-%s-%s -type f' % (eps_,e_d_,e_t_,w_d_))[1].split('\n'))
			print n_files
			if n_files!=3864:
				print 'Running SaveShotsBlocks %s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)
				subprocess.call(["python","SaveShotsBlocks.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
		
		num_cores = 7
		slices = partition(cats,num_cores)
		pool = multiprocessing.Pool(num_cores)
		results = pool.map(run_a_slice, slices)
		pool.close()
			