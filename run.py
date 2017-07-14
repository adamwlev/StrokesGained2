import json
from produceDifficulty import doit

n_folds = 15
for eta in [.7,.3,.05]:
    result = doit(eta,n_folds,False,False,False,False,False,False)
    with open('r%g-0.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,False,False,False,False,False)
    with open('r%g-1.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,True,False,False,False,False)
    with open('r%g-2.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,True,True,False,False,False)
    with open('r%g-3.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,True,False,True,False,False)
    with open('r%g-4.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,True,False,True,True,False)
    with open('r%g-5.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))

    result = doit(eta,n_folds,True,True,False,True,True,True)
    with open('r%g-6.json' % eta,'w') as json_file:
        json_file.write(json.dumps(result))