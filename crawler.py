import subprocess, gc

if __name__=="__main__":
	e_d = '0.8'
	e_t = '0.7'
	w_d = '0.8'
	alpha = '0.85'
	beta = '8'
	block_size, window_size = '6', '18'

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']
		
	for cat in cats:
		subprocess.call(["python","produce_skill_estimates.py" ,cat,e_d,e_t,w_d,alpha,beta,block_size,window_size])
		gc.collect()