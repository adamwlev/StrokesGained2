import subprocess
import sys
import os
import itertools

if __name__=="__main__":
	
	eps = ['300']
	e_d = ['0.01','0.23','0.66']
	e_t = ['0.01','0.15','0.4']
	w_d = ['0.3','0.7','0.95']
	alpha = ['0.8','0.93','0.99']
	beta = ['5','9','13','17']

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

	done_cats,done_ranks = {},{}
	if not os.path.exists('outFile.csv'):
		outFile = open('outFile.csv','w')
	else:
		with open('outFile.csv','r') as f:
			for line in f.read().splitlines():
				if line.stripe().split('-')[-1]=="cats":
					done_cats.add(tuple(map(float,line.strip().split('-')[:-1])))
				else:
					done_ranks.add(tuple(map(float,line.strip().split('-')[:-1])))
		outFile = open('outFile.csv','a')

	for eps_,e_d_,e_t_,w_d_,alpha_,beta_ in itertools.product(*[eps,e_d,e_t,w_d,alpha,beta]):
		if tuple(map(float,[eps_,e_d_,e_t_,w_d_])) not in done_cats:
			subprocess.call(["python","SaveTheCats.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
			outFile.write("%s-%s-%s-%s-%s-%s-cats" % (eps_,e_d_,e_t_,w_d_))

		if tuple(map(float,[eps_,e_d_,e_t_,w_d_,alpha_,beta_])) not in done_ranks:
			for cat in cats:
				if not os.path.exists('./../ranks/ranks_%s-%s-%s-%s-%s-%s/%s_ranks.npy' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_,cat)): 
					subprocess.call(["python","SaveTheRanks.py" ,"%s" % cat,"%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_, "%s" % alpha_, "%s" % beta_])
					subprocess.call(["rsync","-avL","--progress","-e",'"ssh',"-i",'/home/ubuntu/aws_ds8key.pem"',
									 "/home/ubuntu/project/Rank_a_Golfer/ranks/ranks-%s-%s-%s-%s-%s-%s" % (eps_,e_d_,e_t_,w_d_,alpha_,beta_),
									 "ubuntu@ec2-52-23-248-152.compute-1.amazonaws.com:~/project/Rank_a_Golfer/ranks/"])
			outFile.write("%s-%s-%s-%s-%s-%s-ranks" % (eps_,e_d_,e_t_,w_d_,alpha_,beta_))
			outFile.write("\n")

	    	