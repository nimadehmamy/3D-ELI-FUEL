#!/usr/bin/env python

from numpy import logspace
import os,sys

if len(sys.argv)<2:
    print "run_all_th5.py base_file [0,1(GPU)] [{params_update}]"
    exit()
    
base_file = sys.argv[1] #'../../output/phase-100917/ER-0-n70-L136.json'
f = lambda s: int(base_file.split('.json')[0].split('-'+s)[1].split('-')[0]) 
n,l = f('n'), f('L')
C = (int(sys.argv[2]) if len(sys.argv)>2 else 1) 
kw = (eval(sys.argv[3]) if len(sys.argv)>3 else {})

rs = lambda n,l: n**(1./3)*l**(-.5)
x0 = rs(n,l)

# xi = x0*logspace(-3,3,20)
# xi = x0*logspace(-1.2,1.2,10)
# xi = x0*logspace(1.3,3.1,4)

# xi = x0*logspace(-3,-1.3,10)
# xi = x0*logspace(1.3,2.5,5)

N= 6 #10
if len(sys.argv)>4:
    b = int(sys.argv[4])
    if b==0:
        xi = x0*logspace(-3,-1.,N)
    elif b==1:
        xi = x0*logspace(-1.,1.,N)
    elif b==2:
        xi = x0*logspace(1.,2.5,N)
else:
    # xi = x0*logspace(-2.5,2.5,20)
    # xi = x0*logspace(-1.5,1.5,15) #25)#,15)
    xi = x0*logspace(-1.2,1.2,25) #25)#,15)
    # xi = x0*logspace(-1.1,1.1,20)#,15)
    # xi = x0*logspace(-2.5,2.5,15)
    
    # xi = x0*logspace(-1,1,15) #,20)
    # xi = x0*logspace(-2.5,2.5,16)
    

for th in xi:
    #os.system('CUDA_VISIBLE_DEVICES=%d ./run_sims5.py %s %.3g "%s" &' %(C,base_file, th, kw))
    #os.system('CUDA_VISIBLE_DEVICES=%d ./run_sims6.py %s %.3g "%s" &' %(C,base_file, th, kw))
    #os.system('CUDA_VISIBLE_DEVICES=%d ./run_sims7.py %s %.3g "%s" &' %(C,base_file, th, kw))
    # 042518
    os.system('CUDA_VISIBLE_DEVICES=%d ./run_sims8.py %s %.3g "%s" &' %(C,base_file, th, kw))
    