from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pyGAT.utils import load_data, accuracy
from pyGAT.models import GAT


class fitModel():
    def __init__(self,model,optimizer,fastmode):
        # self.utils = utils.GeneralUtils()
        # self.check = utils.ExaminationalUtils()
        # self.model_utils = utils.ModelUtils()
        # self.SMO_utils = SourceMutationOperatorsUtils()
        self.model=model
        self.optimizer=optimizer
        self.fastmode=fastmode
        pass

    def train(self, epoch, all_features, idx_train, idx_val, adj, labels):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(all_features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        if not self.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(all_features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))


    def test(self,all_features, idx_test, adj,labels):
        self.model.eval()
        output = self.model(all_features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        preds = output[idx_test].max(1)[1].type_as(labels[idx_test])

        correct = preds.eq(labels[idx_test]).double()
        acc_test = correct.sum()/ len(labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), preds

    # def accuracy(self, output, labels):
    #     preds = output.max(1)[1].type_as(labels)
    #     correct = preds.eq(labels).double()
    #     correct = correct.sum()
    #     return correct / len(labels)

    def train_and_test(self, all_features, adj, idx_train, idx_val, idx_test, labels, epochs):
        # Train model (Original)
        for epoch in range(epochs):
            self.train(epoch, all_features, idx_train, idx_val, adj, labels)
        # print("Optimization Finished!")

        # Testing (Original)
        tst_accuracy, predictions = self.test(all_features, idx_test, adj,labels)
        # print("Testing Finished!")

        return tst_accuracy, predictions, self.model
