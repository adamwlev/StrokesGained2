import subprocess
import sys
import os
import itertools
import commands

if __name__=="__main__":
	
	eps = ['300']
	e_d = ['0.01','0.23','0.66']
	e_t = ['0.01','0.15','0.4']
	w_d = ['0.3','0.7','0.95']
	alpha = ['0.8','0.93','0.99']
	beta = ['5','9','13','17']

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

	for eps_,e_d_,e_t_,w_d_,alpha_,beta_ in itertools.product(*[eps,e_d,e_t,w_d,alpha,beta]):
		if not os.path.exists('./../cats/cats_w%s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)):
			print 'Running SaveTheCats %s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)
			subprocess.call(["python","SaveTheCats.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
		else:
			n_files = len(commands.getstatusoutput('find ./../cats/cats_w%s-%s-%s-%s -type f' % (eps_,e_d_,e_t_,w_d_))[1].split('\n'))
			print n_files
			if n_files!=3808:
				print 'Running SavetheCats %s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)
				subprocess.call(["python","SaveTheCats.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
		for cat in cats:
			if not os.path.exists('./../ranks/ranks-%s-%s-%s-%s-%s-%s/%s_ranks.npy' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_,cat)): 
				print 'Running SaveTheRanks %s-%s-%s-%s-%s-%s %s' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_,cat)
				subprocess.call(["python","SaveTheRanks.py" ,"%s" % cat,"%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_, "%s" % alpha_, "%s" % beta_])
				cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/ranks/ranks-%s-%s-%s-%s-%s-%s ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/ranks/" % (eps_,e_d_,e_t_,w_d_,alpha_,beta_)
				os.system(cmd)