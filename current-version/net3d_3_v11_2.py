# Double-Ellipsoid forces
# v10
# Limit total dx using arcsinh
# many networks (especially small ones) easily become unstable
# we have to control how far each element can be shot 
# v10.1
# !!! get_ct changed:
# r^2/DL --> r/DL
# also, now use second derivative to rescale c0 to c1
# fixed weighted bug for netRads and fixed minor ct issues in iter_converge
# v10.2 
# fixed unweighted link bug 
# v10.3
# !!! reverting to older get_ct:
# new one with log ratios for m and s was unstable. 
# !!! Reverting to one with different r dimensionality
# v10.4
# making POW... input parameters for networkBase
# 
# v10.6
# !!! reverting to newer get_ct with log ratios
# check_convergence: removing the "or (abs(a1.mean) < .5*tol)" which overrid other conditions
# check_convergence: adding the "or (abs(a1.mean) < .05*tol)" which overrid other conditions
# now the convergence should be stricter and more accurate. 
# !!! fixed bug in ellipsoid: dr2**POW --> dr2**(POW/2.)

# v10.8
# netRads accepting node and link labels
# !!!! bugfix on networkBase get_json:
# new params were ignored. Now JSON updates params before making network.


#import tensorflow as tf
from numpy import *
import json,os,time
from warnings import showwarning
try:
    import tensorflow as tf
except ImportError:
    showwarning("Can't import tensorflow!", ImportWarning, 'net3d_v9_4_10',8)

try:
    from matplotlib.pyplot import *
    #from matplotlib.mlab import movavg
except ImportError:
    showwarning("Can't import matplotlib.pyplot!", ImportWarning, 'net3d_v9_4_10',8)

############################    
__version__ = 'v11.1'
__doc__ = """
Binning both link and node repulsive forces. 
"""
############################

PAIRS = 100
POW = 2
POWn = 2
POW_SN = 2 # power extra factors of r_i+r_j in node repulsion
#pairs = array([i for i in itertools.combinations(arange(400),2)])

def expand(pts, n):
    s = pts.shape
    x = linspace(0,s[0]-1,n)
    xp = arange(s[0]) 
    return array([interp(x,xp,i) for i in pts.T]).T


def dvecz(x,y):
    #assert (len(x.shape)==2) and (x.shape[1] == y.shape[1])
    xT_ = tf.constant(x[:,:,newaxis].T, dtype=tf.float32)
    y_ = tf.constant(y[:,:,newaxis], dtype=tf.float32)
    delta_v = tf.transpose(xT_-y_, perm = [2,0,1])#.shape
#     init = tf.global_variables_initializer()
#     sess.run(init)
    return delta_v

def vec_len(r,ep = 1e-9):
    return ep+tf.sqrt(tf.reduce_sum(tf.pow(r,2),-1)) #[:,newaxis]


class tictoc():
    def __init__(self):
        self.prev = 0
        self.now = 0
    def tic(self):
        self.prev = time.time()
    def toc(self):
        self.now = time.time()
        print( "dt(s) = %.3g" %(self.now - self.prev))
        self.prev = time.time()
        
tt = tictoc()


def Lattice(n, d=3):
    m = int(n**(1./d))+1
    print( m) 
    pts = array([float32((arange(n)/m**i) % m) for i in range(d)]).T
    edg = []
    for i in range(10):
        for j in range(i+1,n):
            if absolute(pts[i]-pts[j]).sum() == 1:
                edg += [[i,j]]
#     for i in range(n):
    return pts,edg


########################
# Simple Layout, for fast initial layout
########################
class netSimple:
    def __init__(self,pts, edg, directed = False, **kw):
        self.pts = array(pts)
        self.elist = array(edg)
        self.get_Adj_mat(directed)
        self.params = {
            'A': 200,
            'pow': POWn,
            'n_radius': 1.,
            'k': 10,
        }
        self.params.update(kw)
        
        self.dt = 1e-3
        self.it_num = 0
        self.tv = []
        self.t = 0
        self._init(kw)
        
    def _init(self):
        pass 
    
    def get_Adj_mat(self, directed):
        self.Adj = zeros((len(self.pts), len(self.pts)))
        for l in self.elist:
            w = (l[2] if len(l) > 2 else 1.)
            l0 = tuple(int0(l[:2]))
            self.Adj[l0] =  w
            if not directed:
                self.Adj[l0[::-1]] = w 
            
    def iter_(self):
        self.get_distz()
        self.t += self.dt
        self.tv += [[self.t, self.lens()]]
        self.it_num += 1

        self.compute_forces()
        self.f_max = abs(self.force_mat).max()
        self.update_dt()
        self.pts += self.dt * self.force_mat.sum(0)
        
    def iter_all(self,t, ct = 0.1):
        self.ct = ct
        for i in range(t):
            self.iter_()
            
        
    def update_dt(self):
        """should be done repeatedly, but as it converges exponentially, we'll use it once in every iter."""
        if self.dt > self.ct/self.f_max:
            self.dt /= 2
        elif self.dt < 2*self.ct /self.f_max:
            self.dt *= 2
            
    def compute_forces(self):
        self.comp_F_NN()
        self.comp_F_NL()
        self.force_mat = self.F_NN + self.F_NL
        
    def get_distz(self):
        self.dp = (self.pts - self.pts[:,newaxis])
        self.dpn = linalg.norm(self.dp, axis=-1, keepdims=1)
        
    def lens(self):
        return (self.dpn[:,:,0] * self.Adj).sum()/2
        
    def comp_F_NN(self):
        A = self.params['A']
        ex = self.params['pow']
        r = self.params['n_radius']
        self.F_NN = A * self.dp / r * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        
    def comp_F_NL(self):
        k = self.params['k']
        self.F_NL = - k * self.dp * self.Adj[:,:,newaxis] # weights also determine strength of springs
        

class netRads(netSimple):
    def _init(self,kw):
        self.params['nodes']={
            'radius':self.params['n_radius'],
            'weighted':True,
            'labels':range(len(self.pts))
        } # for compatibility with networkBase
        self.params.setdefault('long_rep', 0.01) 
        self.params['links']= {
            'T0': self.params['nodes']['radius'] / 2.,
            'ce': 100.,
            'weighted':True,
        }
        if 'nodes' in kw: self.params['nodes'].update(kw['nodes'])
        if 'links' in kw: self.params['links'].update(kw['links'])
        
        self.gnam = 'net-n%d-L%d' %(len(self.pts),len(self.elist))
        self.get_degrees()
        self.set_radii()

    def get_degrees(self):
        self.degrees = self.Adj.sum(1)
    
    def set_radii(self):
        d = self.pts.shape[-1]
        rs = lambda x: ((x**(d-1)).sum(-1))**(1./(d-1))+1e-5 # use root sum squared of weights as radii for 3D
        self.degreesRS = r0 = (rs(self.Adj) if self.params['nodes']['weighted'] else array([1]*len(self.Adj)))
        
        self.rad_mat = self.params['nodes']['radius']*(r0+r0[:,newaxis])[:,:,newaxis]
        
    def comp_f_short(self):
        A = self.params['A']
        ex = self.params['pow']
        r = self.rad_mat # self.params['n_radius']
        #self.f_short = A * self.dp / r * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        # we want forces to get stronger for larger nodes because they also have more elastic forces pulling them 
        self.f_short = A * self.dp * r**POW_SN * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        # this should automatically solve any issue from strong elastic forces
        
    def comp_f_long(self):
        d = float(self.pts.shape[-1])
        self.f_long = self.params['long_rep'] * self.params['A'] * self.dp / (self.dpn**(d-1) +1e-3)**(d/(d-1))
        
    def comp_thermal_noise(self):
        self.noise = self.params['links']['T0'] * exp(- self.it_num/self.params['links']['ce']) * random.randn(*self.pts.shape)
        
    def iter_(self):
        r = self.params['nodes']['radius']/10.
        self.get_distz()
        self.t += self.dt
        self.tv += [[self.t, self.lens()]]
        self.it_num += 1

        self.compute_forces()
        self.f_max = abs(self.force_mat).max()
        self.update_dt()
        self.pts += r * arcsinh( self.dt * self.force_mat.sum(0) / r) 
        
        
    def iter_all(self,t,ct=0.1, **kw):
        self.ct = ct
        for i in range(t):
            self.iter_()
            self.comp_thermal_noise()
            self.pts+= self.noise
        
    def comp_F_NN(self):
        self.comp_f_short()
        self.comp_f_long()
        self.comp_thermal_noise()
        
        self.F_NN = self.f_short + self.f_long 
        
    def make_links(net):
        #['end_points', 'points', 'radius', 'weight']
        lnks = {}
        nl = net.params['nodes']['labels']
        for k in net.elist:
            k0 = int0(k[:2])
            lnks[str('%s;%s'%(nl[k0[0]],nl[k0[1]]))] = { #str(k0)
                'end_points': k0.tolist(), 
                'points': net.pts[k0].tolist(), 
                'radius': net.params['links']['thickness']*(1 if len(k)<3 else k[2]), 
                'weight': 1
            }
        return lnks

    def net_json(net):
        net.JSON = {'scale':1,
                   'links': net.make_links(), #net.make_links(),
                   'nodes': {
                       'positions':net.pts.tolist(),
                       'labels': net.params['nodes']['labels']#range(len(net.pts))
                   },
                   'info': net.params
                   }
        #net.JSON['info']['nodes'].update({'labels':net.JSON['nodes']['labels']})

    def save(net, path, tv=True):    
        fnam = path+'/'+net.gnam
        net.params['links']['thickness'] = .1* net.params['nodes']['radius'] 
        net.net_json() #net.net_json()
        json.dump(net.JSON, open(fnam+'.json', 'w'))
        if tv: savetxt(fnam+'-tv.txt',net.tv) 

        
        
def draw_net_tv_old(nn,dims = [0,1], figsize=(10,5), node_col = '#3355ba', link_col = '#33aa55', 
                new=True, tv=True, **kw):
    simple = not hasattr(nn, 'links')
    pts = (nn.pts if simple else nn.nodes.points.eval() )
    
    if new: figure(figsize=figsize)
    subplot(121, aspect = 'equal')

    scatter(*pts.T[dims],zorder=100, c=node_col)
    if simple:
        for l in nn.elist:
            plot(*nn.pts[int0(l)[:2]].T[dims],c=link_col)
    else:
        l = nn.links.points.eval()
        for i in nn.links.idx.values():
            plot(*l[i[0]:i[1]+1,dims].T, c=link_col,  marker='', ls = '-')#, c = 'r')
    
    if tv:
        subplot(122)
        if tv > 1: 
            loglog(*array(nn.tv[-int(tv):]).T)
        else: 
            loglog(*array(nn.tv).T)

def draw_net_tv(nn,dims = [0,1], figsize=(10,5), node_col = '#3355ba', link_col = '#33aa55', 
                new=True, tv=True, draw_num='all',**kw):
    if draw_num == 'all':
        draw_num = -1
    simple = not hasattr(nn, 'links')
    pts = (nn.pts if simple else nn.nodes.points.eval()[:draw_num] )
    
    if new: figure(figsize=figsize)
    subplot(121, aspect = 'equal')

    scatter(*pts.T[dims], zorder=100, c=node_col)
    if simple:
        for l in nn.elist:
            plot(*nn.pts[int0(l)[:2]].T[dims],c=link_col)
    else:
        l = nn.links.points.eval()
        ii = 0
        for i in nn.links.idx.values():
            plot(*l[i[0]:i[1]+1,dims].T, c=link_col,  marker='', ls = '-')#, c = 'r')
            ii+= 1; 
            if (ii>draw_num) and (draw_num > 0): break
    
    if tv:
        subplot(122)
        if tv > 1: 
            loglog(*array(nn.tv[-int(tv):]).T)
        else: 
            loglog(*array(nn.tv).T)


def net_prog(net,steps,**kw):# ct=0.1, **kw):
    draw_net_tv(net,node_col = '#e82a62', link_col = '#e88f2b', tv=False, **kw )
    tt.tic()
    net.iter_all(steps,**kw)# ct=ct, **kw)
    tt.toc()
    draw_net_tv(net, new = 0, **kw)
    show()
    
def get_dl(self, win = 'all'):
    """compute abs. gradient of mean link length in log time"""
    if win =='all':
        n = len(self.tv)
    else: n = win
    g0 = gradient(log(self.tv[-n:]))[0].T
    g1 = g0[1]/g0[0]
    g2 = gradient(g0[1])/g0[0]**2
    return arcsinh(g1) ,arcsinh(g2)
    
def check_convergence(self, tol = .01, win = 500, **kw):
    a1,a2 = get_dl(self,win = win)
    ch = ((a2.mean() > 2*tol) or (abs(a2.mean()) < tol )) and (a1.mean() < 0 )
    l = len(a1)
    ch2 = (a1[:int(l/2)].mean() < 0) * (a1[int(l/2):].mean() < 0)
    ch3 = (a1[:int(l/2)].mean() > -tol) * (a1[int(l/2):].mean() > -tol)
    
    return (ch*ch2*ch3) or (abs(a1.mean()) < .5*tol), a1.mean(), (ch,ch2,ch3)
        
def run(self,its, draw,**kw): # ct,
    if draw:
        return net_prog(self,steps=its, **kw) # ct=ct,
    else:
        return self.iter_all(its,**kw) # ct=ct,

def iter_converge(self,its = 'auto', max_its = 8e4, draw=True, verbose=True, eps = 0.1, save_path =None, save_its = False, **kw):
    # the average space available per link
    simple = not hasattr(self, 'links')
    degrees = (self.degrees if simple else self.nodes.degrees) 
    r3 = mean((self.params['nodes']['radius']*degrees)**3)**(1./3) # / self.pts.shape[0]
    gnam0 = self.gnam + ''
    if its=='auto':
        # run 10x the thermal time, till the noise dies
        its = max(int(10*self.params['links']['ce']),100)
    if simple and 'ct' not in kw:
        run(self,its, draw, ct = eps*r3,**kw)
    else:
        run(self,its, draw,**kw)
        
    print ("Beginning convergence check ...")
    ck = False
    while (self.it_num < max_its) and not ck:
        if save_path: 
            if not save_its: self.save(save_path,tv=1)
            else: 
                self.gnam = gnam0 + '-it' + str(self.it_num)
                self.save(save_path,tv=0)
        self.params['long_rep'] = 1e-3 + 100./ (self.it_num+1) 
        ct = (eps*max([r3 , self.pts.std()/self.pts.shape[0]**(1./3)]) + 10./(self.it_num+1) if simple else 'auto')
        if simple and 'ct' not in kw:
            run(self,its, draw, ct = ct, **kw)
        else: 
            run(self,its, draw, **kw)
        ck,gm,ch = check_convergence(self,win = its, **kw)
        print( '\n%d) ct=%.2g, '% (self.it_num, (ct if simple else self.ct) ) \
        + (self.gnam if not simple else '')\
        + (' ' if ck else ' Has Not ') + 'Converged! dlog<l>/dlog(t)~%.3g' %(gm), "\nChecks:", ch )

        
class networkBase():
    def __init__(self, pts=[], edg=[],fixed=False, JSON=None, keep_paths=False, 
                 PAIRS = PAIRS, POW =POW , POWn =POWn , POW_SN =POW_SN, # power extra factors of r_i+r_j in node repulsion
                 max_workers = 20, **kw):
        """ 
        pts: node positions
        edg: edge list
        fixed: if nodes should remain fixed
        JSON: if network should be loaded from a json file (specify file paths or file object)
        keep_paths: whether to keep original link trajectories inside JSON file
        max_workers: # of parallel units for foce calculations
        kw: 
        links = {'k': <spring constant>, 
                'thickness': <# or list of thicknesses>,
                'amplitude': <of repulsive force>,
                'ce': <cooling exponent for sim. annealing>,
                'Temp0': <initial temperature as a fraction of thickness>,
                'segs': <# of segments on each link>}
        nodes = {'amplitude': <of repulsive force>,
                'radius': <range of repulsive Gaussian force>}
        net.points contians all link points. The link_link interaction matrix
        should now be part of the net.links object, not the individual links. This allows massive
        vectorization of all interactions.
        
        net.links:
            Now we have a single net_links object that contains info of all links.
            All that separates them is the net_links.idx dict which indexes which points in
            net_links.points belong to which link.
            
            net_links also contains all methods for interactions between links.
        """
        tt.tic()
        # All calcs done in self.session
        self.session = tf.InteractiveSession()
        # All dynamical equations kept in assignments
        self.assignments_forces = tuple()
        self.assignments_points = tuple()
        
        
        self.params = {
            'links':
                   {'k':1e1,'amplitude':5e2, 'thickness':.1, 
                    'Temp0':.5, 'ce':1000, 'segs':5,
                   'weighted': 1,
                   },
            'nodes':
                   {'amplitude' : 5e2, 
                    'radius': 1., 
                    'weighted':1},
                  }
        
        
        self.gnam = 'E-ELF-sim'
        
        self.dt0 = 1e-5 #tf.Variable(1e-5, dtype=tf.float32)
        self.dt = tf.Variable(1e-5, dtype=tf.float32)
        self.dt_ = tf.placeholder(tf.float32)
        
        if JSON:
            self.get_JSON(JSON,kw)

        self.it_num = 0
        kwl = (kw['links'] if 'links' in kw else {})
        kwn = (kw['nodes'] if 'nodes' in kw else {})
        self.params['links'].update(kwl)
        self.params['nodes'].update(kwn)
        
        self.max_workers = max_workers
        self.fixed = fixed
        self.keep_paths = keep_paths
        self.PAIRS = PAIRS
        self.params['POW'] = self.POW = POW
        self.params['POWn'] = self.POWn = POWn
        self.params['POW_SN'] = self.POW_SN = POW_SN
        
        if not JSON:
            self.pts = array(pts) #tf.Variable(pts, dtype=tf.float32)
            self.elist = array(edg,dtype=int32)[:,:2]  #tf.Variable(edg, dtype=tf.float32)
            self.link_weights = (array(edg)[:,2] if self.params['links']['weighted'] else array([1]*len(self.elist)))
            if 'labels' in self.params['nodes']:
                self.make_link_labels()
                             
        
        
        self.it_num = 0
        self.t = 0
        self.tv = [] # to save volume evolution
        # initialize variables
        self.th_mean = mean(self.params['links']['thickness']* self.link_weights )
        self.nrad_mean = mean(self.params['nodes']['radius'])
        self.r_min = float32(min(self.nrad_mean, self.th_mean)/10.)
        self.f_mild = lambda x: self.r_min * tf.asinh( x / self.r_min )
        
        # for assignments
        self.asg = {}
                 
        tt.toc(); print("Making links...")
        self.links = NetLinks(net = self, **self.params['links'])#(**kwl)
        tt.toc(); print( "Making nodes...")
        self.nodes = NetNodes(net = self, **self.params['nodes'])#(**kwn)
        tt.toc(); print( "initializing global variables...")
        init = tf.global_variables_initializer()
        self.session.run(init)
        tt.toc(); print( "Initial binning...")
        self.it_updates()
        tt.toc(); print( "setup: dt...")
        self.setup_dt()
        tt.toc(); print( "setup: volume...") 
        self.vol = tf.reduce_sum(vec_len(self.links.dp)) 
        tt.toc(); print( "setup: dynamics: forces...",) 
        # Define the comp group and iteraion steps
        self.step_forces = tf.group(*self.assignments_forces)
        tt.toc(); print( "setup: dynamics: points, dt...") 
        self.step_points = tf.group(*self.assignments_points)
        self.step_dt = tf.group(*self.asg['dt'])
        tt.toc(); print( "Done!" )
        
    def get_JSON(self,JSON,kw):
        """when extra kw provided, JSON must be updated"""
        if type(JSON) == dict:
            js = JSON
        elif type(JSON) == str:
            js = json.load(open(JSON, 'r'))
        else:
            raise TypeError
        
        #self.params['info'] = js['info']
        self.params.update(js['info']) 
        # update params before proceeding, otherwise only json params are used.
        kwl = (kw['links'] if 'links' in kw else {})
        kwn = (kw['nodes'] if 'nodes' in kw else {})
        self.params['links'].update(kwl)
        self.params['nodes'].update(kwn)
 
        self.params['nodes']['labels'] = list(js['nodes']['labels'])
        self.pts = array(js['nodes']['positions'])
        self.elist = [] 
        if ('labels' not in self.params['links']): 
            self.params['links']['labels'] =[]
            
        self._lnk_points0 = {}# []
        self._lnk_idx0 = {}
        seg = 0
        #self.params['links']['thickness'] = []
        self.link_weights = [] # now thicknesses are inside link weights
        # self.params['links']['thickness'] : just a numerical factor
        wg = (self.params['links']['weighted'] if 'weighted' in self.params['links'] else False)
        for i, k in enumerate(js['links']):
            dc = js['links'][k]
            self.params['links']['labels'] +=[k]
            self.elist += [dc['end_points']]
            self.link_weights += [(dc['radius'] if wg else 1)]
            #self.params['links']['thickness'] += [dc['radius']]
            p = dc['points']
            self._lnk_points0[tuple(dc['end_points'])] = array(p)
            self._lnk_idx0[i] = [seg, seg + len(p)-1]
            seg += len(p)
        self.elist = array(self.elist)
        self.link_weights = array(self.link_weights)
      
    def get_neighbors(self,v):
        '''get neighbor list of node v'''
        return [tuple(i) for i in self.elist if v in i]

    def setup_dt(self):
        self.ct_ = tf.placeholder(tf.float32)
        #fn = (self.f_node_max *(1.+1e-5 -1.*(self.fixed))) # if not self.fixed else 1.)
        mx = array(minimum(self.nrad_mean,self.th_mean),dtype=float32)
        self.asg['dt'] = (self.dt.assign( 
            self.ct_*tf.minimum( mx,
                tf.minimum (self.th_mean/self.f_link_max, self.nrad_mean/self.f_node_max))),)
                 
    # def get_ct(self, w= 200, c0 = .1, eps_ct = 1e-2, **kw):
    #     i = array(self.tv[-w:])
    #     if len(i) < w:
    #         self.ct = c0
    #         return 0
    #     # measure gradient of std
    #     s = i[:,1].std()
    #     # compare with trend, if almost monotonic
    #     s1 = abs(i[-w/10:,1].mean()-i[:w/10,1].mean())
    #     # The ratio should be the required ct
    #     self.ct = eps_ct+ nan_to_num(c0*max([self.th_mean,self.nrad_mean])**2 /(s1+s+1e-6)) #c0*s1/s + eps
    
    # def get_ct(self, w= 200, c0 = .5, eps_ct = 1e-2, **kw):
    #     i = array(self.tv[-w:])
    #     if len(i) < w:
    #         self.ct = c0
    #         return 0
    #     # measure gradient of std
    #     s = i[:,1].std()
    #     # compare with trend, if almost monotonic
    #     w0 = w/5#max(int(w/sqrt(w)),1)
    #     i = array(self.tv[-w:])[:,1]
    #     ibeg, iend = i[:w0].mean(), i[-w0:].mean()
    #     m = abs(iend-ibeg)
    #     gbeg, gend = i[w0:2*w0].mean()-i[:w0].mean(), i[-w0:].mean()- i[-2*w0:-w0].mean() 
    #     # If change in grad is small, increase ct factor
    #     c1 = c0 * max(abs((gbeg+gend)/2/(gend-gbeg)),1e-3)
    #     # The ratio should be the required ct
    #     self.ct = eps_ct+ nan_to_num(c1 * max([self.th_mean,self.nrad_mean])**1 /(m+s+1e-6)) #c0*s1/s + eps
    
    def get_ct(self, w= 200, c0 = .5, eps_ct = 1e-4, **kw):
        if len(self.tv) < w:
            self.ct = c0
            return 0
        i = array(self.tv[-w:])[:,1]
        # measure gradient of std
        s = i.std()/ i.mean()
        # compare with trend, if almost monotonic
        w0 = int(w/5)#max(int(w/sqrt(w)),1)
        # make everything scale independent 
        
        ibeg, iend = i[:w0].mean(), i[-w0:].mean()
        m = abs(iend-ibeg)/i.mean()
        gbeg, gend = i[w0:2*w0].mean()-i[:w0].mean(), i[-w0:].mean()- i[-2*w0:-w0].mean() 
        # If change in grad is small, increase ct factor
        self.dg = max(abs((gbeg+gend)/2/(gend-gbeg)),1e-2)
        c1 = c0 * self.dg
        # The ratio should be the required ct
        # now, c1, m and s are all scel-independent
        self.ct = eps_ct+ nan_to_num(c1 /(m+s+1e-6)) #c0*s1/s + eps
        self.ct = arcsinh(self.ct)
    
       
    def it_updates(self,ct = 'auto', rebin = 'auto' , bin_size = 10, **kw ):
        if rebin == 'auto':
            if self.it_num >= self.links.Bins.next_binning:
                self.links.Bins.auto_rebin(min_cells = self.max_workers, **kw)
                self.nodes.Bins.auto_rebin(min_cells = self.max_workers, **kw)
                print( self.it_num, end=' ')
                if ct == 'auto': 
                    self.get_ct(**kw)
                    print( '|ct:%.2g'%self.ct, end=' ')
                else: self.ct = ct
        else: 
            if self.it_num % rebin == 0:
                self.links.Bins.bin_rand(bin_size = bin_size)
                self.links.Bins.update_bin_keys()
                
                self.nodes.Bins.bin_rand(bin_size = bin_size)
                self.nodes.Bins.update_bin_keys()
                
                print( self.it_num, end=' ')

    
    def iter_all(self,it = 1, ellipsoid = False, **kw):# rebin = 'auto' , bin_size = 10, ct='auto',  **kw):
        """Calls
        1) all force functions in all nodes and links to update their forces.
        2) Updates dt (not doing so before moving results in unstable dynamics)
        3) Then, it calls iter_dyn functions to update positions.
        """ 
        for i in arange(it):
            self.it_updates(**kw)#rebin = rebin, bin_size = bin_size, ct = ct)          
            self.step_forces.run()
            if not self.fixed:
                self.nodes.Force_repel.iter_full()            
            if ellipsoid:
                self.links.Force_ext_ell.iter_full()
            else: 
                self.links.Force_ext.iter_full()
            self.step_dt.run({self.ct_:self.ct})
            self.step_points.run({self.ct_:self.ct,
                                  self.links.noise_: self.links.thermal_noise()})
            
            self.it_num += 1
            self.t += self.dt.eval()
            self.tv+=[[self.t,self.vol.eval()]]
          
        
    def make_link_labels(self):
        """Use if nodes and edges are given as arrays, and node labels are given separately."""
        n = self.params['nodes']['labels']
        self.params['links']['labels'] = ['%s; %s'%(n[i],n[j]) for i,j in self.elist]
        
    def make_JSON(self):
        self.netJSON ={'nodes':{}, 'edgelist':[], 'links':{}, 'info':self.params}
        self.netJSON['name'] = self.gnam
        pts_now = self.nodes.points.eval() 
        self.netJSON['nodes']['positions'] = pts_now.tolist()
        nd = self.params['nodes']
        self.netJSON['nodes']['labels']=(nd['labels'] if 'labels' in nd else
                                         [i for i in range(len(self.netJSON['nodes']['positions']))] )
        ld = self.params['links']
        self.netJSON['links']['labels']=(ld['labels'] if 'labels' in ld else
                                         [i for i in range(len(self.elist))])
        th = self.links.thickness[::self.links.segs].tolist() #self.params['links']['thickness']
        links = {}
        lnk = self.links.points.eval()
        ix = self.links.idx
        for i in ix:
            l = self.elist[i] #
            dat = lnk[ix[i][0]:ix[i][1]+1]
            lab = self.netJSON['links']['labels'][i] 
            #('%d,%d'%tuple(l) if self.labels == 'auto' else self.labels[i])
            links[lab]={ # i labels the multi-link
                    'end_points': l.tolist(),
                    'points': dat.tolist(),
                    'radius': (th[i] if type(th)==list else th),
                    'weight': 1
                               }
        self.netJSON['scale'] = 1.
        self.netJSON['links'] = links
            
    def save(self, path='./', tv=True):
        path += '/'#('/' if path[-1] is not '/' else '')
        p0 = path+self.gnam
        
        self.make_JSON()
        name = self.netJSON['name'] #+ ('/' if self.netJSON['name'][-1] is not '/' else '')
        json.dump(self.netJSON, open(path+name+'.json','w'))
        if tv: savetxt(path+name+'-tv.txt',self.tv)
            
            
class NetNodes():
    """Node class:
    1) position index in net.points
    2) edgelist
    3) uses a dict of link objects and node objects to calculate forces.
    """
    def __init__(nodes,net, **kw):
        """pos=[x,y,...], elist=list of 2-tuples of edges."""
        nodes.net = net
        for k,v in kw.items(): # add rest of params as attributes
            setattr(nodes, k,v)
        nodes.make_nodes(**kw)
        # create degree list
        # print(type(nodes.labels))
        nodes.get_degreesRS()
        nodes.Bins = bin_pts(nodes, unit = max(nodes.degrees), split = 1)
        nodes.force_end()
        nodes.dyn_setup()
        if not nodes.net.fixed:
            nodes.get_r0()
            nodes.Force_repel = BinnedCompute(nodes,nodes.force_node_repel_idx,nodes.net.max_workers)
        
    def get_r0(nodes):
        nodes.r0 = float32((nodes.degrees**1. if nodes.weighted else ones_like(nodes.degrees))\
                           *nodes.radius)  
        

    def get_degreesRS(nodes):
        rs = lambda x : sqrt((array(x)**2).sum(-1))
        nodes.deg_dict = {}
        for i,w in zip(nodes.net.elist, nodes.net.link_weights):
            nodes.deg_dict.setdefault(nodes.labels[i[0]],[])
            nodes.deg_dict.setdefault(nodes.labels[i[1]],[])

            nodes.deg_dict[nodes.labels[i[0]]] += [(w if nodes.weighted else 1)]
            nodes.deg_dict[nodes.labels[i[1]]] += [(w if nodes.weighted else 1)]
        nodes.degrees = array([(rs(nodes.deg_dict[i]) if i in nodes.deg_dict else 0.001) for i in nodes.labels])

    def get_degrees(nodes):
        nodes.deg_dict = {}
        for i,w in zip(nodes.net.elist, nodes.net.link_weights):
            nodes.deg_dict.setdefault(nodes.labels[i[0]],0)
            nodes.deg_dict.setdefault(nodes.labels[i[1]],0)

            nodes.deg_dict[nodes.labels[i[0]]] += (w if nodes.weighted else 1)
            nodes.deg_dict[nodes.labels[i[1]]] += (w if nodes.weighted else 1)
        nodes.degrees = array([nodes.deg_dict[i] for i in nodes.labels])

    def dyn_setup(nodes):
        if not nodes.net.fixed:
            nodes.net.asg['NL_F'] = (nodes.net.f_node.assign(nodes.Force_End),)
            nodes.net.assignments_forces += nodes.net.asg['NL_F']
            nodes.net.f_node_max = tf.reduce_max(tf.abs(nodes.net.f_node))
            #*tf.to_float(not nodes.net.fixed) +1e-4 
            nodes.dp = nodes.net.f_mild( nodes.net.f_node * nodes.net.dt )
        else:
            nodes.net.f_node_max = 1e-4
        nodes.net.asg['N'] = ( (nodes.points.assign_add( nodes.dp),) 
                            if not nodes.net.fixed else tuple() )
        l = [] # pairs of node, link indices 
        for i in range(len(nodes.net.pts)):
            ni = nodes.link_ends[i]
            l+= zip([i]*len(ni),ni)
        l = array(l)
        n2 = tf.gather(nodes.points, l[:,0])
        nodes.net.asg['NL'] = (tf.scatter_update(nodes.net.links.points, l[:,1], n2),)
        nodes.net.assignments_points += nodes.net.asg['N'] + nodes.net.asg['NL']

    def make_nodes(nodes, regions =1,**kw):
        # nodes.points will be used in iter dyn. ==> must be tensorflow
        nodes.points = tf.Variable(nodes.net.pts, dtype=tf.float32)
        nodes.net.f_node = tf.Variable(0*nodes.net.pts, dtype=tf.float32)
        # 1) nodes.elist: links attached to each node (by their index in edg)
        # 2) nodes.link_ends: which end of link attached to node
        # both info in net.elist, so one loop over net.elist is enough
        nodes.elist = {n: [] for n in range(len(nodes.net.pts))}
        nodes.link_ends = {n: [] for n in range(len(nodes.net.pts))}
        for i, (n1,n2) in enumerate(nodes.net.elist):
            nodes.elist[n1] += [i]
            nodes.elist[n2] += [i]
            # index of links.points attached to n1 and n2:
            nodes.link_ends[n1] += [nodes.net.links.idx[i][0]]
            nodes.link_ends[n2] += [nodes.net.links.idx[i][1]]
        if not hasattr(nodes, 'labels'):
            nodes.labels = range(len(nodes.net.pts))
    
    def force_node_repel_idx(nodes,idx):
        """try out different forces, long- and short-range"""
        r0 = tf.gather(nodes.r0, idx)
        th = r0 + r0[:,newaxis]

        A = nodes.amplitude
        x = tf.gather(nodes.points, idx)
        r = x - x[:,newaxis,:] #tf.expand_dims(x,1)
        rlen = vec_len(r)
        #fmat = A*r*tf.expand_dims((rlen/th)**(POWn-2)*tf.exp(-(rlen/th)**POWn),2)
        #fmat = th[:,:,newaxis]**POW_SN *A*r*tf.expand_dims((rlen/th)**(POWn-2)*tf.exp(-(rlen/th)**POWn),2)
        fmat = th[:,:,newaxis]**nodes.net.POW_SN *A*r*tf.expand_dims((rlen/th)**(nodes.net.POWn-2)*tf.exp(-(rlen/th)**nodes.net.POWn),2)

        nodes.Force_NN = tf.reduce_sum(fmat,0)
        return tf.scatter_add(nodes.net.f_node,idx, nodes.Force_NN)
        
    def force_end(nodes):
        """using node positions and elist (a list of neighbors),
        get the link internal forces (int_force) that pull the nodes"""
        nodes.Force_End = tf.concat([[tf.add(array([0,0,0], dtype=float32),
            tf.reduce_sum([ nodes.net.links.Force_Int[j] for j in nodes.link_ends[k]], 0))]
                                      for k in range(len(nodes.net.pts))],0)
        
        
class NetLinks():
    def __init__(links, net=None , **kw):
        """Accesses list of all node positions and the edgelist of net.
        self.points: all link segment positions
        self.idx: dict of all indices for where each link begins and ends in self.points"""
        links.net = net
        links.noise_ = tf.placeholder(tf.float32)
        links.params = kw
        for k,v in kw.items(): # add rest of params as attributes
            setattr(links, k,v)
        # initial thickness * weights
        if links.weighted:
            links.thickness = links.thickness*links.net.link_weights
            links.k = links.k*links.net.link_weights
            
        links.make_attr_pts('thickness')
        #links.thickness_ = tf.constant(links.thickness, tf.float32)
        links.make_attr_pts('k')
        links.make_points()
        # initial temperature
        links.T0 = links.Temp0*links.thickness #mean(links.thickness)
        N,dim = links.points0.shape
        #links.Pair_count = []
        links.force_internal()
        links.Bins = bin_pts(links, unit = max(links.thickness), split = links.segs)

        links.Force_ext = BinnedCompute(links, links.forces_ext_brute_idx_multi,
                                         links.net.max_workers)
        links.setup_v_rot_()
        links.Force_ext_ell = BinnedCompute(links, links.force_ext_ellipsoid_idx_multi,
                                             links.net.max_workers)
        links.dyn_setup()

    def make_attr_pts(links, atr, dtype = float32):
        try:
            if len(getattr(links,atr)) == len(links.net.elist):
                print( atr+" per edge given.")
                # generate full thicknesses for all link points
                th1 = []
                for i in range(len(links.net.elist)):
                    th1 += [getattr(links,atr)[i]]*links.segs
                th1 = array(th1, dtype=dtype)
                setattr(links,atr, th1)
        except TypeError:
            print( "single "+atr+"? Adapting to edge segments...")
            setattr(links,atr, array([getattr(links,atr)]*len(links.net.elist)*links.segs, dtype=dtype))

    def make_points(links):
        links.idx = {}
        links.points0 =[]
        for n,i in enumerate(links.net.elist):
            segs = links.segs #10 #min(100,max(10,segs)) # constrian number of segs
            links.idx[n] = [len(links.points0),len(links.points0)+segs-1]
            # append the segment points of link to net.points
            if links.net.keep_paths: 
                pts = expand(links.net._lnk_points0[tuple(i)], segs)
            else:
                pts = array([linspace(*j,num= segs) for j in links.net.pts[i].T]).T
            if len(links.points0) == 0 :
                links.points0 = pts
            else:
                links.points0 = concatenate((links.points0,pts), axis = 0)
        links.points = tf.Variable(links.points0, dtype=tf.float32)
        links.net.f_link = tf.Variable(0*links.points0, dtype=tf.float32)

    def lens(links,i):
        """all segment lengths for edge i = (n1,n2)"""
        ii = links.idx[i]
        pts = links.points[ii[0]:ii[1]]
        dl = pts[1:]-pts[:-1]
        return vec_len(dl)

    def force_internal(links):
        """Spring forces. For efficiency, also caculates unnecessary, non-existent forces of
        endpoints of different links.
        Beware of that!!"""
        end_pts = array([i for i in links.idx.values()], dtype=int32) # to accelerate correcting endpoint forces
        # first calculate all delta p's, even wrong one from one link to another
        s = links.points0.shape
        links.end_mask = ones((s[0], s[1]),dtype=float32)
        links.end_mask[end_pts[:,1]] *= 0 
        links.end_mask_full = ones((s[0], s[1]),dtype=float32)
        links.end_mask_full[end_pts[:,0]] *= 0 
        links.end_mask_full[end_pts[:,1]] *= 0
        links.dp0 = links.points[:-1] - links.points[1:]
        dp = links.end_mask[:-1] * links.dp0 
        links.dp = dp
        # end_mask to kill cross link subs
        dpt = tf.concat([zeros((1,links.points0.shape[1])),-dp],0)  # left side
        dpt1 = tf.concat([-dp,zeros((1,links.points0.shape[1]))],0) # left springs
        links.Force_Int = links.k[:,newaxis] * links.segs * (dpt1 - dpt)
        # return self.Force_Int

    def setup_v_rot_(links):
        r = links.thickness[:-1,newaxis]
        dl = (links.points[:-1] - links.points[1:]) / 2
        dlh = dl / (tf.norm(dl,axis = 1)[:,newaxis]+1e-10)
        idx = tf.argmin(dlh,1)
        r0 = tf.one_hot(idx,3)
        r1h = tf.cross(dlh, r0)
        r2h = tf.cross(dlh,r1h)
        r1 = r*r1h
        r2 = r*r2h
        vi = tf.concat((dl[:,:,newaxis], r1[:,:,newaxis],r2[:,:,newaxis]),axis = 2)
        ri = tf.concat((dlh[:,:,newaxis], r1h[:,:,newaxis],r2h[:,:,newaxis]),axis = 2)
        links.v_rot = tf.matmul(vi,ri, transpose_b=True )
        # shape is (dl, dim, dim) matrix product for all dl vectors
        # note: this must produce V R^T. It does so by mimicking matrix product. Double-check! 
    # now we have the orientation tensors for every segment of the links. Next, we make the metric for all
    # pairs of segments. It will be a (dl x dl x dim x dim) tensor.

    def get_metrics(links,idx):
        v_rot_idx = tf.gather(links.v_rot, idx)
        # all possible seg pairs
        V = v_rot_idx[newaxis] + v_rot_idx[:,newaxis] 
        eps = 1e-9
        eye_eps = -eps*tf.eye(links.points0.shape[-1],batch_shape = (1,1), dtype = tf.float64)
        ginv = tf.to_double(tf.matmul(V,V,transpose_b=True)) + eye_eps
        gij = tf.to_float(tf.matrix_inverse(ginv))
        return gij, ginv

    def get_dr2(links,r, gij):
        dr = (r[newaxis] - r[:,newaxis])/2
        return tf.matmul(dr[:,:,newaxis],tf.matmul(gij,dr[:,:,:,newaxis]))[:,:,0]
        
    def force_ext_ellipsoid_idx_multi(links,idx):
        gij, ginv = links.get_metrics(idx)
        r = tf.gather(links.points, idx)
        dr2 = links.get_dr2(r,gij)
        A = links.amplitude
        dr = r - r[:,newaxis]
        drh = dr / (tf.norm(dr, axis = -1, keepdims=True)+1e-15)
        #links.fmat0 = A*drh*((links.dr2[id]**(links.net.POW/2-1))*tf.exp(-links.dr2[id]**links.net.POW))
        links.fmat0 = A*drh*((dr2**(links.net.POW/2.-1))*tf.exp(-dr2**(links.net.POW/2.)))
        links.Force_LL_Ell = tf.reduce_sum(links.fmat0,0)
        return tf.scatter_add(links.net.f_link,idx, links.Force_LL_Ell)

#         #### Cos theta 
#         dx = tf.gather(links.dp, idx)
#         dx1 = dx/tf.norm(dx, axis = -1)[:,newaxis]
#         dx2 = tf.matmul(dx1, tf.transpose(dx1))
#         links.cos[id] = dx2 # tf.norm(fmat, axis = -1)
#         #links.fmat[id] = tf.norm(links.fmat0, axis = -1)

    def dyn_setup(links):
        '''define one step of dynamics'''
        # don't move endpoints...
        links.net.asg['LF_Int'] = (links.net.f_link.assign(links.Force_Int),)
        links.net.assignments_forces += links.net.asg['LF_Int']

        links.net.f_link_max = tf.reduce_max(tf.abs(links.net.f_link * links.end_mask_full)) 
        f = links.end_mask * (links.net.dt*(links.net.f_link) + links.noise_)

        links.net.asg['L'] = (links.points.assign_add( links.net.f_mild(f) ),)
        links.net.assignments_points += links.net.asg['L']

    def thermal_noise(self):
        Temp = self.T0*exp(-self.net.it_num/self.ce)
        noise =Temp[:,newaxis]*random.randn(*self.points0.shape)
        return noise
    
    def forces_ext_brute_idx_multi(links, idx):
        """ Generate extra tensors for link external force calculation.
        it has a placeholder links.f_ext_idx_[id] for indexing,
        and defines a tensor links.force_ext_app[id], both unique to this each instance,   
        """
        x = tf.gather(links.points, idx)
        # all possible seg pairs
        th0 = tf.gather(links.thickness, idx)
        th_mat = th0 + th0[:,newaxis]
        A = links.amplitude
        links.r = x - x[:,newaxis]
        rlen = vec_len(links.r)
        # !!! must exclude pairs on same edge, otherwise edge won't contract
        fmat = A*links.r*((rlen/th_mat)**(links.net.POW-2)/th_mat\
                          *tf.exp(-(rlen/th_mat)**links.net.POW)\
                          *links.link_self_mask_multi(idx))[:,:,newaxis]
        # including selfrepulsion again
        # fmat = A*links.r*((rlen/th_mat)**(POW-2)/th_mat*tf.exp(-(rlen/th_mat)**POW))[:,:,newaxis] 

        links.Force_LL = tf.reduce_sum(fmat,0)
        return tf.scatter_add(links.net.f_link,idx, links.Force_LL)
        
    def link_self_mask_multi(links, idx):
        """Create a mask to eliminate self-repulsion
        for the active link indices passed to placeholder links.force_idx_"""
        i1 = tf.floor(tf.to_float(idx / links.segs))
        return tf.to_float(tf.not_equal(i1 - i1[:,newaxis], 0))

class bin_pts():
    def __init__(self, obj, unit = 1, split = 1):
        self.obj = obj
        self.unit = unit
        self.split = split
        self.curr_keys = []
        self.bin_size = 10 # initial default factor
        self.next_binning = 0
        
        self.prep_bins()
        #self.bin_rand()
        
    def prep_bins(self):
        """bin_size: in units unit
        """
        self.bin_size_ = tf.placeholder(tf.float32)
        l0 = self.bin_size_ * self.unit
        l = tf.minimum((tf.reduce_max(self.obj.points) - tf.reduce_min(self.obj.points))/3, l0)
        self.cell_size = l 
        self.cells_shift_ = tf.placeholder(tf.float32, shape=(self.obj.points.shape.as_list()[1],))
        self.cells_raw = tf.to_int32(tf.floor(self.obj.points[:-1]/l +self.cells_shift_))
        # cells_raw : array of int coords for points
        self.bins = {}

    def bin_rand(self, bin_size = 10.): # split = links.segs 
        d = self.obj.points.shape.as_list()[1]
        x = self.cells_raw.eval({self.cells_shift_: random.randn(d), self.bin_size_: bin_size})
        cells = self.hash_(x)
        self.make_bins(cells)
        # self.update_bin_keys()
        
    def hash_(self,x):
        # make a simple code for hashing the corrdinates (faster than string or tuple)
        d = x.shape[-1]
        xm = 2*int(abs(x).max())
        c = [xm**i for i in range(d)]
        cells = (x * c).sum(1) # hash number as keys for cells
        return cells
    
    def make_bins(self,cells):
        """ Go through the indices, make a dict of seg_nums, 
        they already contain info on the link they belong to
        """
        self.bins = {}
        self.bin_set = {} # to keep # diff. links in cells
        for i,v in enumerate(cells):
            self.bins.setdefault(v,[])
            self.bins[v]+=[i]
            self.bin_set.setdefault(v,set())
            self.bin_set[v].add(int(i/self.split))
               
    def update_bin_keys(self):
        self.curr_keys = []
        for i in self.bins:
            if len(self.bin_set[i]) >1: # more than one link in cell
                self.curr_keys +=[i]
    
    def auto_rebin(self, min_cells = 5, # self.links.max_workers
              min_step = 40, **kw): #,c0 = 0.03
        """
        just make sure the number of bins is larger than the min_cells
        if not, rebin with smaller bins
        """
        for i in range(5): # at most 5 times
            mx = max(1,0,*[len(self.bins[k]) for k in self.curr_keys])
            if len(self.curr_keys) < min_cells or mx > PAIRS: # too many points in bins
                print( "^%.2g" %(self.bin_size), end=' ')
                if self.bin_size > 8: # only make smaller if cell_size > 8*unit
                    self.bin_size /= 2.
                else: print( "[=]", end=' ')
                self.bin_rand(self.bin_size)
            elif mx < PAIRS and self.bin_size < 20:
                print ("v%.2g" %(self.bin_size), end=' ')
                self.bin_size *= 2.
                self.bin_rand(self.bin_size)
            else:
                self.bin_rand(self.bin_size)
                self.update_bin_keys()
                break
            print (len(self.curr_keys), end=' ')
            self.update_bin_keys()
        # now we have enough bins, 
        # rebin sooner, if most bins are full, i.e. if len(self.links.bins) ~ len(self.curr_keys)
        b,c = len(self.bins), len(self.curr_keys)
        self.next_binning += abs((10*(b-c))/(c+1)) + min_step 

        
class BinnedCompute:
    def __init__(self,obj,method,num = 20):
        self.obj = obj
        self.max_workers = num
        self.make_indices()
        self.get_tensors(method) # get tensors to compute
        
    def make_indices(self):
        self.idx_ = {i: tf.placeholder(tf.int32) for i in range(self.max_workers)}
            
    def get_tensors(self,method):
        """Takes idx placeholder and returns a tensor"""
        self.tensors = [method(i) for i in self.idx_.values()]
   
    def iter_full(self):
        m = self.max_workers
        for i in range(0, len(self.obj.Bins.curr_keys), m):
            ix = self.obj.Bins.curr_keys[i:i+m]
            m0 = len(ix)
            self.obj.net.session.run(self.tensors[:m0], 
                feed_dict ={self.idx_[i]: self.obj.Bins.bins[v]\
                            for i,v in enumerate(ix)})
   