import subprocess
import sys

if __name__=="__main__":
	_,epsilon,e_d,e_t,w_d = sys.argv

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']
	
	for a in ['0.7','0.8','0.9']:
		for beta in ['8','10','12']:
			for cat in cats:
				subprocess.call(["python","SaveTheRanks.py" ,cat,"%s" % epsilon,"%s" % e_d,"%s" % e_t,"%s" % w_d, "%s" % a, "%s" % beta])