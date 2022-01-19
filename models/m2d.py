import sys

sys.path.extend(['../'])
import models.tools as tools

num_node = 17#num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15)] #AIST大数据集的节点连接顺序
#[(3,2),(4,3),(5,4),(6,5),(8,7),(9,8),(10,9),(11,10),(13,12),(14,13),(15,14),(17,16),(18,17),(19,18), (20,19),(7,2),(12,2),(16,2)]#DWM数据集
inward = [(i, j) for (i, j) in inward_ori_index]#正序连接边
outward = [(j, i) for (i, j) in inward]#逆序连接边
neighbor = inward + outward#表示所有的连接边


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)