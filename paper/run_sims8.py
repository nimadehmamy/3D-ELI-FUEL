#!/usr/bin/env python

#from net3d_v9_4_6 import *
#from net3d_v9_4_8 import *
# from net3d_v9_4_12 import *
# from net3d_v10_1 import *
# from net3d_v10_3 import *
# from net3d_v10_4 import *
# from net3d_v10_6 import *
from net3d_v10_6 import *

import sys 

fnam = sys.argv[1] 
pth = fnam[:-fnam[::-1].find('/')]

xi = float(sys.argv[2])

k,A = .1, 1.* max(xi,1.) #10.* max(xi,1.) #10.*xi #1.*xi #10.* max(xi,1.) #.1/xi ,1.
params = {
         "fixed" : 0, 
         "links" : {'Temp0':.5, 'ce': 10.,
                    'thickness': xi, 'segs': 15,#30,#15, 
                    'k': k,
                    'amplitude': A,#1e2*A,#1e1*A,
                    'max_workers': 70, #100, #70,#150, #70, 
                    'weighted':0}, 
         "nodes" : {
                    'radius': 1.,
                    'weighted': 0,
                    'amplitude': 20*A, #1.*A, #10*A, #4*A,
                 }
         }

pl = params['links']
pn = params['nodes']
kw = {'links':{},'nodes':{}}
if len(sys.argv) > 3:
    kw.update(eval(sys.argv[3]))
params.update(kw)
pl.update(kw['links'])
pn.update(kw['nodes'])
params['links'].update(pl)
params['nodes'].update(pn)

fx = params['fixed']

nn2 = networkBase(
                  JSON=fnam,
                  keep_paths=1,
                 **params) # 50/xi
n,m = nn2.pts.shape[0], nn2.elist.shape[0]

k = params['links']['k']
A = params['links']['amplitude']

nn2.gnam = fnam.split('/')[-1].split('-n')[0]+'-n%d-L%d-th-r%.3g-segs%d-ka%.3g' %(n,m,
            nn2.links.thickness.mean()/nn2.nodes.radius,nn2.links.segs,k/A)\
        +('-fixed' if fx else '')
#nn2.gnam0 = nn2.gnam +''

w = 3000 #2000 #5000 
# iter_converge(nn2, max_its = 6e4, draw=0, save_path = pth)#,its = w, c0=0.01)#,tol=0.1)
# iter_converge(nn2, max_its = 6e4, draw=0, save_path = pth)#, c0=0.01)#,its = w, c0=0.01)#,tol=0.1)

c0 =  .5 #.2 #.5 #.4 #.3 #.5 # 1/sqrt(n) #.3 #.5
tol = c0/6 #/4 #/ 6 #10 # /4 #/10 #/6#/5 #/6 #/4
iter_converge(nn2, c0 = c0, tol= tol, max_its = 2e4, draw=0, save_path = pth)
# run again to make sure it wasn't stuck
print "final relxation: "+ nn2.gnam
# # iter_converge(nn2, c0 = c0/5, tol= tol/5, max_its = 2e3, draw=0, save_path = pth)
for i in range(3):
    nn2.iter_all(it= int(2e2), c0 = c0/10)
    nn2.save(pth, tv=1)
print "\nDONE: "+nn2.gnam
nn2.save(pth, tv=1)
