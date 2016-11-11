import pandas as pd
import numpy as np
import multiprocessing

master_df = pd.read_csv('master_df.csv')
master_df_broadie = pd.read_csv('master_df_broadie.csv')


cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
        'rough375','fairway0','fairway300','fairway540','bunker']

def run_a_slice(its):
	corrs = []
	inds = range(len(x))
	for _ in xrange(its):
		ch = np.random.choice(inds,len(x))
		corrs.append(np.corrcoef(x[ch],y[ch])[0,1])
	return corrs

cm,l_cm,u_cm,cb,l_cb,u_cb = [],[],[],[],[],[]
for cat in cats:
	x = (master_df['Rating_%s' % cat]-master_df['Field_Strength_%s' % (cat,)]).values
	y = master_df_broadie.Finishing_Pct.values
	cm.append(np.corrcoef(x,y)[0,1]*-1)

	num_cores = multiprocessing.cpu_count()
	slices = [200 for _ in range(num_cores)]
	pool = multiprocessing.Pool(num_cores)
	results = pool.map(run_a_slice, slices)
	pool.close()
	results1 = [item for little_list in results for item in little_list]
	l_cm.append(np.percentile(results1,[(100.0-95)/2,(95+100.0)/2])[0]*-1)
	u_cm.append(np.percentile(results1,[(100.0-95)/2,(95+100.0)/2])[1]*-1)
	
	x = (master_df_broadie['Rating_%s' % cat]-master_df_broadie['Field_Strength_%s' % (cat,)]).values
	cb.append(np.corrcoef(x,y)[0,1]*-1)
	num_cores = multiprocessing.cpu_count()
	slices = [200 for _ in range(num_cores)]
	pool = multiprocessing.Pool(num_cores)
	results = pool.map(run_a_slice, slices)
	pool.close()
	results2 = [item for little_list in results for item in little_list]
	l_cb.append(np.percentile(results2,[(100.0-95)/2,(95+100.0)/2])[0]*-1)
	u_cb.append(np.percentile(results2,[(100.0-95)/2,(95+100.0)/2])[1]*-1)

	print map(len,[cm,l_cm,u_cm,cb,l_cb,u_cb])

d = {'Category':cats,'Correlation with Future Success New Method':cm,' Lower 95% C.I.':l_cm,' Upper 95% C.I.':u_cm,'Correlation with Future Success Strokes Gained to Field Rolling Weighted Average':cb,'Lower 95% C.I.':l_cb,'Upper 95% C.I.':u_cb}
outCoors = pd.DataFrame(d)[['Category','Correlation with Future Success New Method',' Lower 95% C.I.',' Upper 95% C.I.','Correlation with Future Success Strokes Gained to Field Rolling Weighted Average','Lower 95% C.I.','Upper 95% C.I.']]
outCoors.to_csv('outCorrs.csv',index=False)
