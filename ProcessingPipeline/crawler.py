import subprocess
import sys

if __name__=="__main__":

	_,epsilon,e_d,e_t,w_d = sys.argv
	if not os.path.isfile('./../cats/cats_w%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d)):
	    sys.exit('File already exists.')