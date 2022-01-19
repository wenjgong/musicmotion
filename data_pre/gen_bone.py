#Generate the skeleton side of the connection node
import os
import numpy as np
from numpy.lib.format import open_memmap
import tqdm
out_path = '../data/AIST_segment_single'#'../data/MusicToDance'
m2d = ((3,2),(4,3),(5,4),(6,5),(8,7),(9,8),(10,9),(11,10),(13,12),(14,13),(15,14),(17,16),(18,17),(19,18),
          (20,19),(7,2),(12,2),(16,2))
# m2d_seg = ((0,1),(2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),
#           (19,20),(2,7),(2,12),(2,16),(3,7),(3,12),(7,16),(12,16))
aist_m2d = [(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15)]

sets = {
    'train', 'val','test'
}

for set in sets:
    paths = os.path.join(out_path, '{}_joint.npy'.format(set))
    data = np.load(paths)#(20046, 3, 300, 25, 2)

    N, C, T, V = data.shape
    outs = os.path.join(out_path, '{}_bone.npy'.format(set))
    fp_sp = open_memmap(outs,dtype='float32',mode='w+',shape=(N, 3, T, V))
#(20046, 3, 300, 25, 2)
    fp_sp[:, :C, :, :] = data
    for i,(v1, v2) in enumerate(aist_m2d):
        fp_sp[:, :, :, i] = data[:, :, :, v1] - data[:, :, :, v2]#存储v1的节点到v2的节点的长度（v1- v2）(20046, 3, 300, 25, 2)
    print(fp_sp.shape)
