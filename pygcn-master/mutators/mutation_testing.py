from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
from mutators.train_and_test_GNN import fitModel
from mutators.utils_mutator import sample_test_suite



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--mutation_ratio', type=float, default=0.5,
                    help='Mutation_Ratio.')
parser.add_argument('--repetition', type=float, default=5,
                    help='Executions each mutation operator.')
# 'Edge', 'All', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6'
parser.add_argument('--mutation_operator', type=str, default='Node',
                    help='Executions each mutation operator.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data (From Program)
ori_adj, features, labels, idx_train, idx_val, idx_test = load_data()


############# Graph Mutation Operator (Initialization) ###########
# Generate Balanced/Unbalanced Test Suite
test_suite_size = 1000
idx_test_balanced, idx_test_unbalanced = sample_test_suite(labels, idx_test, test_suite_size)


class MutationTest():
    def __init__(self, args):
        self.mutation_ratio = args.mutation_ratio
        self.mutant_number_each_mutator = range(args.repetition)
        self.mutator = [].extend(args.mutation_operator)

    def ErrorRate(self, original_predictions, mutant_predictions):
        same_num = original_predictions.eq(mutant_predictions).double()
        same_prediction_rate = same_num.sum() / len(mutant_predictions)
        error_rate = 1.0 - same_prediction_rate.item()
        return error_rate

    def KilledClass(self, original_predictions, mutant_predictions, tst_label):
        original_predictions = original_predictions.numpy()
        mutant_predictions = mutant_predictions.numpy()
        tst_label = tst_label.numpy()
        idx_1 = np.nonzero(original_predictions == tst_label)[0]  # correct in original model
        idx_2 = np.nonzero(mutant_predictions != tst_label)[0]  # incorrect in mutated model
        killedId = np.intersect1d(idx_1, idx_2)
        killedcls = np.unique(tst_label[killedId])
        if len(killedcls) > 0:
            tag = "Killed !"
        else:
            tag = "Survived !"
        return killedcls, len(killedcls), tag

    def mutation_testing(self, test_suite):
        idx_test=test_suite
        # Original Model Training and Testing
        global features,labels,ori_adj,idx_train,idx_val
        ori_model = GCN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout)
        optimizer = optim.Adam(ori_model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        if args.cuda:
            ori_model.cuda()
            features = features.cuda()
            ori_adj = ori_adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        t_total = time.time()
        Fit_ori = fitModel(model=ori_model, optimizer=optimizer, fastmode=args.fastmode)
        ori_tst_acc, ori_preds, trained_ori_model = Fit_ori.train_and_test(features,
                                                                           ori_adj,
                                                                           idx_train, idx_val, idx_test, labels,
                                                                           args.epochs)
        print("Original graph + balanced testing suite-- testing accuracy is", ori_tst_acc)

        from mutators.graph_mutator_operators import GraphMutationOperators
        graph_mut_operators = GraphMutationOperators()
        normal_accs = []
        mutant_accs = []
        sum_killedclasses = 0.0
        sum_errorrate = 0.0
        mutant_num = 0.0
        ori_features = features



        ############### Mutation Testing ###################
        # Mutator_2 (Add Edges): graph mutants,  train new models, test
        if 'O2' in self.mutator or 'Edge' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.addEdge_mut(ori_adj, self.mutation_ratio)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels, args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_2 (Add Edges) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_3 (Remove Edges): graph mutants,  train new models, test
        if 'O3' in self.mutator or 'Edge' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.removeEdge_mut(ori_adj, self.mutation_ratio)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels, args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_3 (Remove Edges) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_4 (Perturb Edges): graph mutants,  train new models, test
        if 'O4' in self.mutator or 'Edge' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.perturbEdge_mut(ori_adj, self.mutation_ratio, mu=0, sigma=2.0)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels, args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_4 (Perturb Edges) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_5 (Edge Direction Conversion): graph mutants,  train new models, test
        if 'O5' in self.mutator or 'Edge' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.convertEdgeOrien_mut(ori_adj, self.mutation_ratio)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels, args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_5 (Edge Direction Conversion) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # # # Mutator_6 (Add Nodes): graph mutants,  train new models, test

        # # for self.mutation_ratio in self.mutation_ratios:
        if 'O6' in self.mutator or 'Node' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj, mutated_feat, mutated_labels, mutated_idx_train = graph_mut_operators.addNode_mut(ori_adj,
                                                                                                               ori_features,
                                                                                                               labels,
                                                                                                               idx_train,
                                                                                                               self.mutation_ratio)
                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(mutated_feat,
                                                                     mutated_adj,
                                                                     mutated_idx_train,
                                                                     idx_val,
                                                                     idx_test,
                                                                     mutated_labels,
                                                                     args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_6 (Add Nodes) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_7 (Remove Nodes): graph mutants,  train new models, test
        if 'O7' in self.mutator or 'Node' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj, mutated_feat, mutated_labels, mutated_idx_train, mutated_idx_val, mutated_idx_test = \
                    graph_mut_operators.removeNode_mut(ori_adj,
                                                       ori_features,
                                                       labels,
                                                       idx_train,
                                                       idx_val,
                                                       idx_test,
                                                       self.mutation_ratio)
                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(mutated_feat,
                                                                     mutated_adj,
                                                                     mutated_idx_train,
                                                                     mutated_idx_val,
                                                                     mutated_idx_test,
                                                                     mutated_labels,
                                                                     args.epochs)
                _, ori_preds_removed = Fit_ori.test(features, mutated_idx_test, ori_adj, labels)
                error_rate = self.ErrorRate(ori_preds_removed, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds_removed, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_7 (Remove Nodes) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_8 (Node Purtubation): graph mutants,  train new models, test
        if 'O8' in self.mutator or 'Node' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.perturbNode_mut(ori_adj, self.mutation_ratio, 0.1)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels,
                                                                     args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_8 (Node Purtubation) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))

        # Mutator_9 (Node Exchange): graph mutants,  train new models, test
        if 'O9' in self.mutator or 'Node' in self.mutator or 'All' in self.mutator:
            for _ in self.mutant_number_each_mutator:
                mutated_adj = graph_mut_operators.ExchangeNode_mut(ori_adj, self.mutation_ratio)

                new_model = GCN(nfeat=features.shape[1],
                                nhid=args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=args.dropout)
                new_optimizer = optim.Adam(new_model.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)
                if args.cuda:
                    new_model.cuda()
                t_total = time.time()
                Fit_mut = fitModel(model=new_model, optimizer=new_optimizer, fastmode=args.fastmode)
                mutant_acc, mutant_preds, _ = Fit_mut.train_and_test(features,
                                                                     mutated_adj,
                                                                     idx_train, idx_val, idx_test, labels,
                                                                     args.epochs)
                error_rate = self.ErrorRate(ori_preds, mutant_preds)
                sum_errorrate += error_rate
                killedclasses, _, kill_or_survive = self.KilledClass(ori_preds, mutant_preds, labels[idx_test])
                sum_killedclasses += len(killedclasses)
                print("Mutator_9 (Node Exchange) | mut_ratio =", self.mutation_ratio,
                      "| Testing mutant: tst_acc = ", mutant_acc,
                      "| Rate of Different Predictions:", error_rate,
                      "| Killed Class (num", len(killedclasses), "):", killedclasses,
                      "\n     ", kill_or_survive)
                mutant_num += 1
                mutant_accs.append(mutant_acc)
                print("......Time elapsed: {:.4f}s".format(time.time() - t_total))


        mutation_score = sum_killedclasses / mutant_num * (labels.max().item() + 1)
        print("Overall Mutation Score:", mutation_score)
        print("Average Error Rate: ", sum_errorrate / mutant_num)
        return  mutation_score, sum_errorrate / mutant_num

    def options(self, mutator_lst):
        self.mutator = mutator_lst


# Mutation Testing
mutator_list=['O2','O3','O4','O5','O6','O7','O8','O9','Node','Edge','All']
mutation_test_obj = MutationTest(args)

# Run on single operators
for opt in mutator_list[0:8]:
    mutation_test_obj.options([opt])
    a, b = mutation_test_obj.mutation_testing(idx_test_balanced)
    aa, bb = mutation_test_obj.mutation_testing(idx_test_unbalanced)
    print("Only Testing Mutation Operator",opt,"and getting mutation score :",[a,aa],"and error rate",[b,bb])

# Run on Edge operators
mutation_test_obj.options(['Edge'])
a, b = mutation_test_obj.mutation_testing(idx_test_balanced)
aa, bb = mutation_test_obj.mutation_testing(idx_test_unbalanced)
print("Only Testing Mutation Operators on Edges, and getting mutation score :",[a,aa],"and error rate",[b,bb])

# Run on Node operators
mutation_test_obj.options(['Node'])
a, b = mutation_test_obj.mutation_testing(idx_test_balanced)
aa, bb = mutation_test_obj.mutation_testing(idx_test_unbalanced)
print("Only Testing Mutation Operators on Nodes, and getting mutation score :",[a,aa],"and error rate",[b,bb])

# Run on ALL operators
mutation_test_obj.options(['All'])
a, b = mutation_test_obj.mutation_testing(idx_test_balanced)
aa, bb = mutation_test_obj.mutation_testing(idx_test_unbalanced)
print("Testing ALL Mutation Operators, and getting mutation score :",[a,aa],"and error rate",[b,bb])
