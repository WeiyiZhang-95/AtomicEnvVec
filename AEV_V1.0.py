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
weight_lambda = 2

_ifWeightByR_ = False
_ifWeightBySinTheta_ = False
_ifWeightByR0_ = False
_ifWeightBySinTheta0_ = False

###This code only works while 2*max(cutoffs) <= min(periods)

file_name = 'min10051'

def findBox(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    box = []
    for line in lines:
        if line[-3:-1] == 'hi':
            box.append(line)
        if box!=[] and line[-3:-1]!='hi':
            break
    for i in range(3):
        box[i] = [float(e) for e in box[i].split()[:2]]
    box = np.array(box)
    box = box[:,1]-box[:,0]
    return box
box = findBox(file_name)

def readAtoms(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    while lines[0][0].isdigit() or ('Atoms' not in lines[0] and 'atoms' not in lines[0]):
        lines.pop(0)
    lines.pop(0)
    new = []
    for line in lines:
        if line[0].isdigit():
            new.append(line)
        elif len(line.split()) == 0:
            continue
        else:
            break
    atoms = [e.split() for e in new]
    df = pd.DataFrame(atoms)
    df.columns = ["idx", "type", "x", "y","z"]
    df['idx'] = [int(e) for e in df['idx']]
    df['type'] = [int(e) for e in df['type']]
    df['x'] = [float(e) for e in df['x']]
    df['y'] = [float(e) for e in df['y']]
    df['z'] = [float(e) for e in df['z']]
    return df
df = readAtoms('min10051')
positions = df[['x','y','z']].to_numpy()


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
for i in set(df['type']):
    with open('process','a+') as f:
        f.write(('type:')+str(i)+('\n'))
    type_idxs.append(np.array(df[df["type"] == i]['idx'])-1)

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


AEV = np.zeros((len(positions),typN,typN,len(MuR)+len(MuS)))

for atom_idx in range(len(positions)):
    for type1 in range(typN):
        d_type1 = [d for d in d_matrix[atom_idx,type_idxs[type1]] if 0 < d <= cutoff_r]
        for idx, r in enumerate(MuR):
            if _ifWeightByR_:
                AEV[atom_idx, type1, :, idx]=sum(gaussian(d_type1, r, sig = SigmaR)/np.array(d_type1))
            else:
                AEV[atom_idx, type1, :, idx] = sum(gaussian(d_type1, r, sig = SigmaR))
                if _ifWeightByR0_:
                    AEV[atom_idx, type1, :, idx]=AEV[atom_idx, type1, :, idx]/r
    if atom_idx%500 == 0:
        with open('process','a+') as f:
            f.write(('radius:>>>>')+str(atom_idx)+('<<<<\n'+time.ctime(time.time())+'\n'))
    if atom_idx == len(positions) - 1:
        with open('process','a+') as f:
            f.write(('radius:>>>>')+str(atom_idx)+('<<<<\n=========='+time.ctime(time.time())+'\n'))

for atom_idx in range(len(positions)):
    for type1 in range(typN):
        for type2 in range(type1+1):
            weights = []
            angles = []
            d_type1 = [[i,d2,d] for i, d2, d in zip(type_idxs[type1],d2_matrix[atom_idx,type_idxs[type1]],d_matrix[atom_idx,type_idxs[type1]]) if 0 < d <= cutoff_theta]
            d_type2 = [[i,d2,d] for i, d2, d in zip(type_idxs[type1],d2_matrix[atom_idx,type_idxs[type1]],d_matrix[atom_idx,type_idxs[type1]]) if 0 < d <= cutoff_theta]
            for atom1_idx, atom1_d2, atom1_d in d_type1:
                for atom2_idx, atom2_d2, atom2_d in d_type2:
                    if atom1_idx == atom2_idx:
                        continue
                    d_12 = d2_matrix[atom2_idx,atom1_idx]
                    #print((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                    theta = np.arccos((-d_12+atom1_d2+atom2_d2)/2.0/atom1_d/atom2_d)
                    angles.append(theta)
                    weights.append(math.exp(-(atom1_d+atom2_d)/weight_lambda))
            for idx_s, l in enumerate(MuS):
                if _ifWeightBySinTheta_:
                    AEV[atom_idx, type1, type2, idx_s+10] = sum(gaussian(angles, l, sig = SigmaS)*np.array(weights)/np.sin(angles))
                else:
                    AEV[atom_idx, type1, type2, idx_s+10] = sum(gaussian(angles, l, sig = SigmaS)*np.array(weights))
                    if _ifWeightBySinTheta0_:
                        AEV[atom_idx, type1, type2, idx_s+10] = AEV[atom_idx, type1, type2, idx_s+10]/math.sin(l)
                AEV[atom_idx, type2, type1, idx_s+10] = AEV[atom_idx, type1, type2, idx_s+10]

    if atom_idx%100 == 0:
        with open('process','a+') as f:
            f.write(('angle:>>>>')+str(atom_idx)+('<<<<\n'+time.ctime(time.time())+'\n'))
    if atom_idx == len(positions) - 1:
        with open('process','a+') as f:
            f.write(('angle:>>>>')+str(atom_idx)+('<<<<\n=========='+time.ctime(time.time())+'\n'))

np.save('AEV.npy',AEV)
