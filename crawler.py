import subprocess, gc

if __name__=="__main__":
	e_d = '0.8'
	e_t = '0.7'
	w_d = '0.8'
	alpha = '0.95'
	beta = '4'
	block_size, window_size = '4', '22'

	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']
		
	for beta in ['4','6','8','10','12','14','16']:
		for cat in cats:
			subprocess.call(["python","produce_skill_estimates.py" ,cat,e_d,e_t,w_d,alpha,beta,block_size,window_size])
			gc.collect()