import subprocess
import sys

if __name__=="__main__":
	_,epsilon,e_d,e_t,w_d = sys.argv

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

	for cat in cats:
		subprocess.call(["python","SaveTheRanks.py" ,cat,"90","0.7","0.25","0.8", "0.93","14"])

	for a in ['0.87','0.97']:
		for beta in ['8','13','16']:
			for cat in cats:
				subprocess.call(["python","SaveTheRanks.py" ,cat,"%s" % epsilon,"%s" % e_d,"%s" % e_t,"%s" % w_d, "%s" % a, "%s" % beta])