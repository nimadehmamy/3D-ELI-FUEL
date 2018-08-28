# 3D Network Layout Algorithm (ELI and FUEL) Python Simulation Code

## Latest version
Software versions used: 
* __Python:__ 3.5.2
* __TensorFlow:__ 1.8.0
* current version __`net3d_3_v11_2.py`__.

TL;DR: Checkout `current-version/3dart-flavor-steps-052718.ipynb` (v11.2); or `3dart-test.ipynb` for a complete set of examples (using v10.8).

__v11.2__: Major changes and restructuring. The new version uses space partitioning for nodes, whereas the old version only did so for link segments. 
This allows networks with thousands of nodes to be simulated in a reasonable amount of time. 
The downside is, in small systems interactions across partition walls may result in perpetual fluctuations, as partitions are moved around randomly to remove boundary effects. 
For small networks it is recommended to use `net3d_3_v10_8.py` instead. 

For a complete set of examples

## Paper simulation code
The code used to generate the simulations used in the paper "Structural Transition in Physical Networks" (Dehmamy et.al.) can be found in the `paper/` directory.  
Software versions used: 
* __Python:__ 2.7.12
* __TensorFlow:__ 1.6.0 

Some examples of how to use the module are given in `3D-TF-auto-1.ipynb` (copy module from `old-src/net3d_v9_4_4.py`). 
The module `net3d_vX_X.py` provides a number of classes and functions for laying out networks in 3D. 
The important layout classes are:
* __`netRads(pts,edg,**kw)`__: A simple layout algorithm with short and long-range repulsion among nodes and no repulsion among links. When the long-range node  repulsion is small and links are thin, `netRads` produces layouts similar to the full algorithm `networkBase`. The main purpose of `netRads` is to run a quick layout algorithm, ignoring link repulsion, to have a good starting point for the full algorithm `networkBase`. The output is a JSON in  a proprietary format described below. Inputs are
    * node positions `pts=np.array([x1,y1,z1]... [xN,yN,zN]])`
    * edgelist `edg=[[n1,n2],...]` (or weighted eidgelist `edg=[[n1,n2,w]...]`, together with `links={'weighted':True,...}` in `**kw`)
    

* __`networkBase(pts=[],edg=[],fixed=False,JSON=None, keep_paths=False,...**kw)`__: The full simulation with elastic, curving links. Either provide: 
    * node positions `pts=np.array([x1,y1,z1]... [xN,yN,zN]])` and
    * edgelist `edg=[[n1,n2],...]` (or weighted edgelist `edg=[[n1,n2,w]...]`, together with `links={'weighted':True,...}` in `**kw`), 
    * Or provide a JSON file created by `networkBase` or `netRads` or manually prepared in the format given below, using `JSON='file.json'`. 
    * `fixed`: Whether to keep nodes fixed (ELI) or allow them to move (FUEL)
    * `keep_paths`: Whether to keep the link trajectories (only if JSON provided). If `False`, will start from straight links. The output can be saved in the JSON format given below.
    * See documentation of `networkBase`, or examples in `run_sims8.py` or the notebooks for description other parameters. 
    
The function `iter_converge(net, max_its, tol... **kw)` is a convenience function which iterates `net` (a `netRads` or `networkBase` object), using their `net.iter_all` method, until it "converges", in the sense that the checks done by `check_convergence(net,tol=tol)` (related to logarithmic changes in total link length) become smaller than a tolerance `tol`.
It also accepts keywords that passed to `net.iter_all` (e.g. `c0` which adjusts the time step scaling can be passed to `iter_converge`; see examples in `run_sims8.py`). 


The output of the simulation can be saved in a proprietary format compatible with [3dviz](https://github.com/nimadehmamy/3dviz) (also accessible [here](http://nimadehmamy.com/3dviz)).
The json contains the node and link segment positions, as well as some of the simulation parameters (`info` key).


```javascript
netJSON = {
    'scale': (size of network. used to shrink down the network to fit in screen.),
    'info': {
            'nodes':{<sim parameters>},
            'links':{<sim parameters>}
            },
    'links':{
            'link1_label': {
                'points': [[xi,yi,zi],...[xf,yf,zf]], //(points along the link 2 minimum)
                'radius': r (thickness of the link),
                    },
            'link2_label': {...},
            ...
            },
     'nodes': {
        'labels':['n0',...,'nN'],
        'positions':[[x1,y1,z1],...[xN,yN,zN]], //(all nodes at once. labels will be supported later.)
        }
```




### Command-line Example
To run a batch of simulations from some base file `input-net.json` (such as the series of simulations used in the paper for the phase diagram) you can run  
`$ ./run_all_th5.py input-net.json 1 "{'links':{'k':.01,'segs':10}}" 1`  

Details of the script are mostly self-explanatory.   

Some of the older versions of the simulation code can be found in `paper/old-src/`. 

For analysis of the phase diagram, see `3D-phase-all-032818.ipynb` (Warning: some plots near the bottom may involve non-converged simulations and dataset not used for the paper. For more details email Nima Dehmamy). 
