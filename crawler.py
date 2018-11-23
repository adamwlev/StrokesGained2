import subprocess, gc

if __name__=="__main__":
    e_d = '0.8'
    e_t = '0.7'
    w_d = '0.8'
    alpha = '0.95'
    block_size, window_size = '4', '22'

    cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
            'rough375','fairway0','fairway300','fairway540','bunker','other']
    beta_map = {'green5':'16',
                'green0':'16',
                'tee45':'4',
                'rough0':'16',
                'fairway540':'16',
                'rough375':'16',
                'green10':'16',
                'other':'10',
                'green20':'16',
                'fairway0':'16',
                'bunker':'16',
                'rough90':'16',
                'fairway300':'16',
                'tee3':'8'}

    for cat in cats:
        subprocess.call(["python3","produce_skill_estimates.py" ,cat,e_d,e_t,w_d,alpha,
                         beta_map[cat],block_size,window_size])
        gc.collect()