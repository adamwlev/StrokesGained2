import subprocess
import sys

if __name__=="__main__":
	_,epsilon,alpha,beta = sys.argv
	epsilon,alpha,beta = tuple(map(float,[epsilon,alpha,beta]))

	# cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
	# 		'rough375','fairway0','fairway300','fairway540','bunker','other']

	cats = ['putting','tee','approach','around_green']
	
	for cat in cats:
		subprocess.call(["python","produceRanks.py" ,cat,"%g" % epsilon, "%g" % alpha, "%g" % beta])