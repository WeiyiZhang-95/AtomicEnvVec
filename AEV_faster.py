__author__ = "Weiyi, Zhang" 
__version__ = "1.0" 

import numpy as np
import pandas as pd
import sys, os
import gc
import math
import time

cutoff_r = 6
cutoff_theta = 6
###This code only works while 2*max(cutoffs) <= min(periods)

with open('min10051','r') as f:
    lines = f.readlines()

atoms = [line.split() for line in lines[26:]]
lines.clear()
df = pd.DataFrame(atoms)
df.columns = ["idx", "type", "x", "y","z"]

df['idx'] = [int(e) for e in df['idx']]
df['type'] = [int(e) for e in df['type']]
df['x'] = [float(e) for e in df['x']]
df['y'] = [float(e) for e in df['y']]
df['z'] = [float(e) for e in df['z']]
positions = df[['x','y','z']].to_numpy()


box = ['3.1781394890167824e+00 4.9274350212712889e+01 xlo xhi\n',
 '3.1781394890167824e+00 4.9274350212712889e+01 ylo yhi\n',
 '3.1781394890167824e+00 4.9274350212712889e+01 zlo zhi\n']
for i in range(3):
    box[i] = [float(e) for e in box[i].split()[:2]]
box = np.array(box)
box = box[:,1]-box[:,0]

def distance(p1,p2=[0,0,0]):
    t = np.array(p1)-np.array(p2)
    return np.dot(t,t)**0.5

v_matrix = np.zeros((len(positions),len(positions),len(positions[0])))
d_matrix = np.zeros((len(positions),len(positions)))
d2_matrix = np.zeros((len(positions),len(positions)))

for i in range(len(positions)):
    v_matrix[i,:,:] = positions[:] - positions[i]
v_matrix = np.absolute(v_matrix)
v_matrix = np.minimum(box - v_matrix, v_matrix)

type_idxs = []
for e in set(df['type']):
    type_idxs.append(np.array(df[df["type"] == e]['idx'])-1)

for i in range(len(positions)):
    for j in range(i):
        d2_matrix[i,j] = d2_matrix[j,i] = np.dot(v_matrix[i,j,:],v_matrix[i,j,:])
        d_matrix[i,j] = d_matrix[j,i] = d2_matrix[i,j]**0.5
    if i%500 == 0:
        with open('process','a+') as f:
            f.write(('distance:>>>>')+str(i)+('<<<<\n')+time.ctime(time.time())+'\n')
    if i == len(positions) - 1:
        with open('process','a+') as f:
            f.write(('distance:>>>>')+str(i)+('<<<<\n=========='+time.ctime(time.time())+'\n'))

np.save('vectors.npy',v_matrix)
np.save('d_matrix.npy',d_matrix)
np.save('d2_matrix.npy',d2_matrix)

typN=len(type_idxs)
MuR=np.linspace(1,6,10)
MuS=np.linspace(0,np.pi,10)
SigmaR=0.3
SigmaS=0.3
import math
def gaussian(x, mu, sig = SigmaS):
    return np.exp( - ( (np.array(x) - mu) / sig)**2 / 2) / (sig*(2*math.pi)**0.5)


AEV = np.zeros((len(positions),typN,typN,len(MuR),len(MuS)+1))

for atom_idx in range(len(positions)):
    for type1 in range(typN):
        for idx, r in enumerate(MuR):
            d_type1 = [d for d in d_matrix[atom_idx,type_idxs[type1]] if d <= r and d != 0.0]
            AEV[atom_idx, type1, :, idx, 0] = sum(gaussian(d_type1, r, sig = SigmaR))
    if atom_idx%500 == 0:
        with open('process','a+') as f:
            f.write(('radius:>>>>')+str(atom_idx)+('<<<<'+time.ctime(time.time())+'\n'))
    if atom_idx == len(positions) - 1:
        with open('process','a+') as f:
            f.write(('radius:>>>>')+str(atom_idx)+('<<<<\n=========='+time.ctime(time.time())+'\n'))

for atom_idx in range(len(positions)):
    for type1 in range(typN):
        for type2 in range(typN):
            angles = []
            pres_r = 0
            pres_d_type1 = []
            pres_d_type2 = []
            for idx_r, r in enumerate(MuR):
                d_type1 = [[i,d2,d] for i, d2, d in zip(type_idxs[type1],d2_matrix[atom_idx,type_idxs[type1]],d_matrix[atom_idx,type_idxs[type1]]) if pres_r < d <= r and d != 0.0]
                d_type2 = [[i,d2,d] for i, d2, d in zip(type_idxs[type1],d2_matrix[atom_idx,type_idxs[type1]],d_matrix[atom_idx,type_idxs[type1]]) if pres_r < d <= r and d != 0.0]
                for atom1_idx, atom1_d2, atom1_d in d_type1:
                    for atom2_idx, atom2_d2, atom2_d in d_type2:
                        d_12 = d2_matrix[atom2_idx,atom1_idx]
                        #print((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        theta = np.arccos((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        angles.append(theta)
                for atom1_idx, atom1_d2, atom1_d in pres_d_type1:
                    for atom2_idx, atom2_d2, atom2_d in d_type2:
                        d_12 = d2_matrix[atom2_idx,atom1_idx]
                        #print((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        theta = np.arccos((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        angles.append(theta)
                for atom1_idx, atom1_d2, atom1_d in d_type1:
                    for atom2_idx, atom2_d2, atom2_d in pres_d_type2:
                        d_12 = d2_matrix[atom2_idx,atom1_idx]
                        #print((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        theta = np.arccos((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                        angles.append(theta)
                pres_d_type1 = d_type1
                pres_d_type2 = d_type2
                pres_r = r
                for idx_s, l in enumerate(MuR):
                    AEV[atom_idx, type1, type2, idx_r, idx_s+1] = sum(gaussian([angle for angle in angles if angle < l], l, sig = SigmaS))

    if atom_idx%100 == 0:
        with open('process','a+') as f:
            f.write(('angle:>>>>')+str(atom_idx)+('<<<<'+time.ctime(time.time())+'\n'))
    if atom_idx == len(positions) - 1:
        with open('process','a+') as f:
            f.write(('angle:>>>>')+str(atom_idx)+('<<<<\n=========='+time.ctime(time.time())+'\n'))

np.save('AEV.npy',AEV)
