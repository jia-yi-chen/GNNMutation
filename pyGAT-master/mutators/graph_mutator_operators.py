import numpy as np
import random
import math
from mutators.utils_mutator import to_sparse, find_nonIntersect
import torch.nn.functional as F
import scipy.sparse as sp
import torch

####################################
# CS 6501 - Software Artifacts
#
# Author: Jiayi Chen (Edge Operators)
#         Jing Ma (Node Operators)
#
# Time:   12/5/2019
#
##################################

class GraphMutationOperators():

    def __init__(self):
        pass


    # Operator-1: Add Edges
    def addEdge_mut(self, adjacent_mat,  mutation_ratio):
        mutated_adj = adjacent_mat.clone()
        mutated_adj=mutated_adj.to_dense()
        N = mutated_adj.shape[0]

        # Get index of 0-value entries (neglect symmetric)
        tmp=torch.triu(torch.ones(N,N))
        no_edge_idx = ((mutated_adj + tmp)==0.0).nonzero()
        perm=torch.randperm(no_edge_idx.shape[0])

        # Generate some position index to be added to Adj
        edge_num = int(N*mutation_ratio)
        print("Randomly Added ",edge_num,"Edges!")
        target_edges = no_edge_idx[perm[:edge_num]]

        # Generate edge
        max_values = torch.max(mutated_adj, 1)[0]
        tmp=torch.zeros(N,N)
        tmp[target_edges[:,0],target_edges[:,1]] += max_values[target_edges[:,0]]
        tmp = tmp + tmp.t()
        mutated_adj += tmp

        # Normalize edges
        mutated_adj = F.normalize(mutated_adj,p=1, dim=1).to_sparse()

        return mutated_adj



    # Operator-2: Remove Edges
    def removeEdge_mut(self, adjacent_mat,  mutation_ratio):
        mutated_adj = adjacent_mat.clone()
        mutated_adj=mutated_adj.to_dense()
        N = mutated_adj.shape[0]

        # Get index of >0-value entries (neglect symmetric)
        tmp = torch.ones(N,N) - torch.triu(torch.ones(N,N))
        having_edge_idx = ((mutated_adj.mul(tmp))> 0.001).nonzero()
        perm=torch.randperm(having_edge_idx.shape[0])

        # Generate some position index to be removed from Adj
        edge_num = int(N*mutation_ratio)
        print("Randomly Removed ",edge_num,"Edges!")
        target_edges = having_edge_idx[perm[:edge_num]]

        # remove edge
        tmp=torch.zeros(target_edges.shape[0])
        mutated_adj[target_edges[:,0], target_edges[:,1]] = tmp
        mutated_adj[target_edges[:,1], target_edges[:,0]] = tmp

        # Normalize edges
        mutated_adj = F.normalize(mutated_adj,p=1, dim=1).to_sparse()

        return mutated_adj

    # Operator-3: Perturb Edges
    def perturbEdge_mut(self, adjacent_mat,  mutation_ratio, mu=0.0, sigma=2.0):
        mutated_adj = adjacent_mat.clone()
        mutated_adj=mutated_adj.to_dense()
        N = mutated_adj.shape[0]

        # Get index of >0-value entries (neglect symmetric)
        tmp = torch.ones(N,N) - torch.triu(torch.ones(N,N))
        having_edge_idx = ((mutated_adj.mul(tmp))> 0.001).nonzero()
        perm=torch.randperm(having_edge_idx.shape[0])

        # Generate some position index to perturb
        edge_num = int(N*mutation_ratio)
        print("Randomly Removed ",edge_num,"Edges!")
        target_edges = having_edge_idx[perm[:edge_num]]

        # Add Gaussian noise to edge weights
        max_values = torch.max(mutated_adj, 1)[0]
        tmp = torch.normal( torch.ones(edge_num) * mu, torch.ones(edge_num) * sigma )
        tmp = tmp.mul(max_values[target_edges[:,0]])
        mutated_adj[target_edges[:,0], target_edges[:,1]] += tmp
        mutated_adj[target_edges[:,1], target_edges[:,0]] += tmp

        # Normalize edges
        mutated_adj = F.normalize(mutated_adj,p=1, dim=1)
        mutated_adj = mutated_adj.to_sparse()

        return mutated_adj

    # Operator-4: Edge Direction Conversion
    def convertEdgeOrien_mut(self, adjacent_mat,  mutation_ratio):
        mutated_adj = adjacent_mat.clone()
        mutated_adj=mutated_adj.to_dense()
        N = mutated_adj.shape[0]

        # Get index of >0-value entries (neglect symmetric)
        tmp = torch.ones(N,N) - torch.triu(torch.ones(N,N))
        having_edge_idx = ((mutated_adj.mul(tmp))> 0.001).nonzero()
        perm=torch.randperm(having_edge_idx.shape[0])

        # Select some edges to change their directions
        edge_num = int(N*mutation_ratio)
        print("Randomly Changed the Directions of ",edge_num,"Undirected Edges!")
        target_edges = having_edge_idx[perm[:edge_num]]

        # Change Direction (LR/RL half-half)
        L_to_R_index_tmp = torch.zeros(target_edges.shape[0])
        L_to_R_index = torch.randperm(edge_num)[:int(edge_num/2)]
        L_to_R_index_tmp[L_to_R_index]=1.0
        R_to_L_index_tmp = torch.ones(target_edges.shape[0])-L_to_R_index_tmp

        mutated_adj[target_edges[:,0], target_edges[:,1]] *= L_to_R_index_tmp
        mutated_adj[target_edges[:,1], target_edges[:,0]] *= R_to_L_index_tmp

        # Normalize edges
        mutated_adj = F.normalize(mutated_adj,p=1, dim=1)
        mutated_adj = mutated_adj.to_sparse()

        return mutated_adj

    # Operator-5: Node add
    def addNode_mut(self, adjacent_mat, features, labels, idx_train, mutation_ratio):
        '''
        Copy a portion of the nodes in the network
        :param adjacent_mat: torch_sparse_tensor, size = [N, N]
        :param features: FloatTensor, size = [N, dx]
        :param mutation_ratio: float
        :return: a mutated feature and adj
        '''
        n = features.shape[0]
        num_new_node = int(mutation_ratio * n)
        copy_node_idx = random.sample(range(n), num_new_node)
        mutated_feat = torch.cat((features, features[copy_node_idx]), dim=0)

        adjacent_mat = adjacent_mat.to_dense()
        mutated_adj = torch.cat((adjacent_mat, adjacent_mat[copy_node_idx]), dim=0)
        mutated_adj = torch.cat((mutated_adj, torch.zeros([n + num_new_node, num_new_node])), dim=1)
        mutated_adj = to_sparse(mutated_adj)

        mutated_labels = torch.cat((labels, labels[copy_node_idx]), dim=0)
        new_idx = torch.tensor(range(n, n + num_new_node))
        mutated_idx_train = torch.cat((idx_train, new_idx), dim=0)

        return mutated_adj, mutated_feat, mutated_labels, mutated_idx_train

    # Operator-6: Node Remove
    def removeNode_mut(self, adjacent_mat, features, labels, idx_train, idx_val, idx_test, mutation_ratio):
        n = features.shape[0]
        num_left_node = n - int(mutation_ratio * n)
        left_node_idx = random.sample(range(n), num_left_node)
        left_node_idx.sort()
        del_node_idx = list(set(range(n)) - set(left_node_idx))
        del_node_idx.sort()
        del_node_idx = torch.tensor(del_node_idx)

        oldToNew_idx = {left_node_idx[i]:i for i in range(len(left_node_idx))}


        left_node_idx = torch.tensor(left_node_idx)
        mutated_feat = torch.index_select(features, 0, left_node_idx)
        mutated_labels = torch.index_select(labels, 0, left_node_idx)

        mutated_idx_train = find_nonIntersect(idx_train, del_node_idx)
        mutated_idx_val = find_nonIntersect(idx_val, del_node_idx)
        mutated_idx_test = find_nonIntersect(idx_test, del_node_idx)

        adjacent_mat = adjacent_mat.to_dense()
        mutated_adj = torch.index_select(adjacent_mat, 0, left_node_idx)
        mutated_adj = torch.index_select(mutated_adj, 1, left_node_idx)

        mutated_idx_train = mutated_idx_train.numpy()
        idx_trn_new = np.zeros_like(mutated_idx_train)
        for i in range(len(mutated_idx_train)):
            old_id = mutated_idx_train[i]
            idx_trn_new[i] = oldToNew_idx[old_id]
        mutated_idx_train = torch.from_numpy(idx_trn_new)

        mutated_idx_val = mutated_idx_val.numpy()
        idx_val_new = np.zeros_like(mutated_idx_val)
        for i in range(len(mutated_idx_val)):
            old_id = mutated_idx_val[i]
            idx_val_new[i] = oldToNew_idx[old_id]
        mutated_idx_val = torch.from_numpy(idx_val_new)

        mutated_idx_test = mutated_idx_test.numpy()
        idx_tst_new = np.zeros_like(mutated_idx_test)
        for i in range(len(mutated_idx_test)):
            old_id = mutated_idx_test[i]
            idx_tst_new[i] = oldToNew_idx[old_id]
        mutated_idx_test = torch.from_numpy(idx_tst_new)

        return mutated_adj, mutated_feat, mutated_labels, mutated_idx_train, mutated_idx_val, mutated_idx_test

    # Operator-7: Node perturbation
    def perturbNode_mut(self, features, mutation_ratio, sigma=0.1):
        n = features.shape[0]
        num_new_node = int(mutation_ratio * n)
        perturb_node_idx = random.sample(range(n), num_new_node)

        features = features.numpy()
        old_features = features[perturb_node_idx]

        noise = np.random.normal(0, sigma, num_new_node * features.shape[1]).reshape((num_new_node, features.shape[1]))
        features[perturb_node_idx] = old_features + noise
        features = torch.from_numpy(features)

        return features

    # Operator-8: Exchange Node features
    def ExchangeNode_mut(self, features, mutation_ratio):
        n = features.shape[0]
        num_new_node = int(mutation_ratio * n)
        if num_new_node % 2 == 1:
            num_new_node += 1

        exchange_node_idx = random.sample(range(n), num_new_node)
        exchange_node_idx_1 = exchange_node_idx[:num_new_node / 2]
        exchange_node_idx_2 = exchange_node_idx[num_new_node / 2:]

        features = features.numpy()
        old_features = np.array(features)

        features[exchange_node_idx_1] = old_features[exchange_node_idx_2]
        features[exchange_node_idx_2] = old_features[exchange_node_idx_1]
        features = torch.from_numpy(features)

        return features
