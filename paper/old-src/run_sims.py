#!/usr/bin/env python

from net3d_v9_4_4 import *
import sys 

#params = json.load(open('./params-BA.json'))

fnam = sys.argv[1] #'../../output/phase-100217/BA-0-n50-L144.json'
xi = float(sys.argv[2])

fx = 0
k,A = .1,1.*xi
params = {"fixed" : fx, 
         "links" : {'Temp0':.0, 'ce': 10.,
                  'thickness':xi,'segs':30, 'k':k,#/xi,
                  'amplitude':1e1*A,
                 'max_workers':30, 'weighted':0}, 
         "nodes": {
                'weighted':0,
                'amplitude':A,
                 }
         }


nn2 = networkBase(
                  JSON=fnam,
                  keep_paths=1,
                 **params) # 50/xi
n,m = nn2.pts.shape[0], nn2.elist.shape[0]
#nn2.it_num = int(fnam.split('-it')[1][:-5])

# nn2.gnam = 'weighted-flavor-lin-n%d-L%d-th-r%.3g-segs%d-ka%.3g' %(n,m,
nn2.gnam = fnam.split('/')[-1].split('-n')[0]+'-n%d-L%d-th-r%.3g-segs%d-ka%.3g' %(n,m,
            nn2.links.thickness.mean()/nn2.nodes.radius,nn2.links.segs,k/A)\
        +('-fixed' if fx else '')
nn2.gnam0 = nn2.gnam +''
#nn2.gnam

nn2.degrees = nn2.nodes.degrees +0.

ep = 0.2
for i in range(2):
    print 'step: ',i,')'
    iter_converge(nn2,its = 2000,draw=0, max_its=50200, tol=1*ep, tv = -400, 
                  save_path= '../../output/phase-100217/',
                  eps = ep, rebin=150)
    #nn2.gnam = nn2.gnam0 + '-it%d'%(nn2.it_num)
    
nn2.save('../../output/phase-100217/', tv=1)
