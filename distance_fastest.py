mport numpy as np
import pandas as pd
import sys, os

cutoff = 6

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

for i in range(len(positions)):
    v_matrix[i,:,:] = positions[:] - positions[i]
v_matrix = np.absolute(v_matrix)
v_matrix = np.minimum(box - v_matrix, v_matrix)


for i in range(len(positions)):
    for j in range(i):
        #d_matrix[i,j] = d_matrix[j,i] = np.dot(v_matrix[i,j,:],v_matrix[i,j,:])**0.5
        d_matrix[i,j] = d_matrix[j,i] = np.dot(v_matrix[i,j,:],v_matrix[i,j,:])
    #if i%500 == 0:
    #    print(i)
np.save('distance.npy', d_matrix)
