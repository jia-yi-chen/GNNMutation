import numpy as np
import scipy.sparse as sp
import torch
import random

####################################
#
# Author: Jiayi Chen
#
# Time:   12/5/2019
#
##################################

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def find_nonIntersect(t1, t2):
    '''
    :return: elements in t1 which are not in t2
    '''
    indices = torch.ones_like(t1, dtype = torch.uint8)
    for elem in t2:
        indices = indices & (t1 != elem)
    intersection = t1[indices]
    return intersection


def sample_test_suite(all_labels, test_idx_list, size):
    '''
    sample a pair of test suites from dataset
    :param labels: ground truth labels
    :param idx_list: idx of test suite
    :param size: size of samples
    '''
    # dict={idx: i for i, idx in enumerate(test_idx_list)}
    dict = {i: idx.item() for i, idx in enumerate(test_idx_list)}
    # tst_labels = all_labels[test_idx_list]#tst_labels[i] denotes the label of node dict{i}
    idx_lb_list = []
    all_labels = all_labels.numpy()
    test_idx_list = test_idx_list.numpy()
    uniq_labels, _ = np.unique(all_labels[test_idx_list], return_inverse=True)
    np.random.shuffle(uniq_labels)

    #  generate the balanced test suite
    # size_per_cls = int(size / len(uniq_labels))-1
    # balanced_test = []
    # for lb in uniq_labels:
    #     idx_lb = np.nonzero(labels == lb)[0]# the index of test data whose label is lb
    #     idx_lb_list.append(idx_lb)
    #     if size_per_cls > len(idx_lb):
    #         size_per_cls = len(idx_lb)
    #     bl_lb = random.sample(list(idx_lb), size_per_cls)
    #     balanced_test.extend(bl_lb)
    # balanced_test.sort()
    # balanced_test=np.array(balanced_test)

    size_per_cls = 1000000
    balanced_test_idx = []
    for lb in uniq_labels:
        mykeys = np.nonzero(all_labels[test_idx_list] == lb)[0]# 0< <total_test_size (not real idx of test nodes)
        idx_lb = [dict[x] for x in mykeys]  # the index of test data whose label is lb
        idx_lb_list.append(idx_lb)
        if size_per_cls > len(idx_lb):
            size_per_cls = len(idx_lb)
    for idx_lb in idx_lb_list:
        # idx_lb=idx_lb_list[uniq_labels[lb]]
        bl_lb = random.sample(list(idx_lb), size_per_cls)
        balanced_test_idx.extend(bl_lb)
    balanced_test_idx.sort()
    balanced_test_idx = torch.from_numpy(np.array(balanced_test_idx))
    print("Labels of balanced test suite ", balanced_test_idx, "is\n",all_labels[balanced_test_idx])

    # generate the imbalanced test suite
    size = len(balanced_test_idx)
    #idx_lb_list = np.array(idx_lb_list)
    #imbalanced_test = idx_lb_list[:size]
    stack_data_by_class = idx_lb_list[0]
    for i, idx_lb in enumerate(idx_lb_list):
    # while len(data_of_class_0) < size:
        if i != 0:
            stack_data_by_class = np.concatenate((stack_data_by_class, idx_lb))
    imbalanced_test_idx = stack_data_by_class[:size]
    imbalanced_test_idx = torch.from_numpy(np.array(imbalanced_test_idx))
    print("Labels of imbalanced test suite ", imbalanced_test_idx, "is\n", all_labels[imbalanced_test_idx])

    # cc = np.unique(all_labels[imbalanced_test_idx])
    return balanced_test_idx, imbalanced_test_idx