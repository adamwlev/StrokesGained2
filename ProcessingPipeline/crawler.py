import subprocess
import sys
import os
import itertools

if __name__=="__main__":
	
	eps = ['300']
	e_d = ['0.01','0.15','0.4','0.75']
	e_t = ['0.01','0.15','0.4']
	w_d = ['0.3','0.7','0.95']
	alpha = ['0.83','0.90','0.94','0.99']
	beta = ['5','9','13','17']

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

	for  eps_,e_d_,e_t_,w_d_,alpha_,beta_ in itertools.product(*[eps,e_d,e_t,w_d,alpha,beta]):
		if not os.path.isfile('./../cats/cats_w%s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_)):
			subprocess.call(["python","SaveTheCats.py","%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_])
		if not os.path.isfile('./../ranks/ranks_%s-%s-%s-%s-%s-%s' % (eps_,e_d_,e_t_,w_d_,alpha_,beta_)): 
			for cat in cats:
				subprocess.call(["python","SaveTheRanks.py" ,"%s" % cat,"%s" % eps_,"%s" % e_d_,"%s" % e_t_,"%s" % w_d_, "%s" % alpha_, "%s" % beta_])
	    	