# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:49:50 2022

@author: assunta ciarlo

check gradient directions

"""
import csv
import numpy as np
from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


file = 'C:/Users/assun/OneDrive/Documenti/GitHub/multishell-qspace-gradients/samples_128_q0.txt'

with open(file,encoding="utf-8") as csvfile: #added encoding format
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    dirs = []
    for row in reader:
        dirs.append(row)

dirs = dirs[22:]
bvals = []

for i,item in enumerate(dirs):
    
    dirs[i] = item[1:]
    bvals.append(int(item[0]))
    
    

fig= plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

dirs = np.array(dirs).astype(float)
ax.scatter(dirs[:,0].ravel(),dirs[:,1].ravel(),dirs[:,2].ravel(),c = bvals)
plt.show()

n = np.linalg.norm(dirs,ord=2,axis=1)
    

#%%

import csv
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

file = 'C:/Users/assun/OneDrive/Documenti/GitHub/multishell-qspace-gradients/siemens_128_q0.dvs'
#file = 'C:/Users/assun/OneDrive/Documenti/GitHub/multishell-qspace-gradients/old_test_siemens/second_35.dvs'
with open(file,encoding="utf-8") as csvfile: #added encoding format
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    dirs = []
    for row in reader:
        dirs.append(row)
        
dirs = dirs[3:]

Vector = [0]*len(dirs)
for bvec in dirs:
    exec(bvec[0])  
    
Vector = np.array(Vector)
idxb = np.where(Vector[:,0]!=0)[0]
fig= plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
n = np.linalg.norm(Vector,ord=2,axis=1)
ax.scatter(Vector[idxb,0].ravel(),Vector[idxb,1].ravel(),Vector[idxb,2].ravel(),c = n[idxb])
plt.show()



#%%

bvecs = np.loadtxt('E:/TEST_PHANTOM_22_11_02-10_53_46-/ENCEFALO_RICERCA_20221102_105420_783000/DTI_TEST_NEW_DIR_0011/DTI_TEST_NEW_DIR_0011_DTI_test_new_dir_20221102105421_11.bvec')
bvals =  np.loadtxt('E:/TEST_PHANTOM_22_11_02-10_53_46-/ENCEFALO_RICERCA_20221102_105420_783000/DTI_TEST_NEW_DIR_0011/DTI_TEST_NEW_DIR_0011_DTI_test_new_dir_20221102105421_11.bval')
fig= plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(bvecs[0,:].ravel(),bvecs[0,:].ravel(),bvecs[0,:].ravel(),c=bvals)
plt.show()