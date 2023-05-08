import os
import os.path as osp
import argparse

import torch
import torch.nn.functional as F

import numpy as np

from torch_geometric.loader import DataLoader

from SGSIB.GNN import GNN
from SGSIB.sub_graph_generator import MLP_subgraph
from SGSIB.utils import train, test, separate_data

from data.creat_dataset import read_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABIDE')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument("--mi_weight", type=float, default=0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument("--pos_weight", type=float, default= 0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--model_learning_rate', type=float, default=0.0005,
                        help='learning rate of graph model (default: 0.0005)')
    parser.add_argument('--SGmodel_learning_rate', type=float, default=0.001,
                        help='learning rate of subgraph model (default: 0.0005)')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    dataset = read_dataset()

    num_node_features = 116
    num_edge_features = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    num_of_fold = 10
    acc_train_list = torch.zeros((num_of_fold,))
    acc_test_list = torch.zeros((num_of_fold,))

    for fold_idx in range(num_of_fold):
        max_acc_train = 0.0
        max_acc_test = 0.0
        
        train_dataset, test_dataset = separate_data(dataset, args.seed, fold_idx)
        
        # Instantiate the backbone network
        model = GIN(num_of_features=num_node_features, device=device).to(device)
        # Instantiate the subgraph generator
        SG_model = MLP_subgraph(node_features_num=num_node_features, edge_features_num=num_edge_features, device=device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.model_learning_rate},
            {'params': SG_model.parameters(), 'lr': args.SGmodel_learning_rate}
            ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for epoch in range(1, args.epochs + 1):
            # Train the model and test it
            avg_loss, mi_loss = train(args, model, train_dataset, optimizer, epoch, SG_model, device)
            acc_train, acc_test, test_loss= test(args, model, train_dataset, test_dataset, SG_model, device)
            
            # print info and save models
            max_acc_train = max(max_acc_train, acc_train)
            acc_train_list[fold_idx] = max_acc_train
            max_acc_test = max(max_acc_test, acc_test)
            acc_test_list[fold_idx] = max_acc_test
            print(f'best accuracy in epoch {epoch} (train / test): ({max_acc_train} / {max_acc_test})')

            savedir = "./SGSIB/model/GCN_model" + str(fold_idx)
            if not osp.exists(savedir):
                os.makedirs(savedir)
            savename = savedir + "/GCN" + "_" + str(epoch) + ".tar"
            torch.save({"epoch" : epoch, "state_dict": model.state_dict(),}, savename)

            savedir = "./SGSIB/model/GCN_model" + str(fold_idx)
            if not osp.exists(savedir):
                os.makedirs(savedir)
            savename = savedir + "/subgraph" + "_" + str(epoch) + ".tar"
            torch.save({"epoch" : epoch, "state_dict": SG_model.state_dict(),}, savename)

            filename="./SGSIB/model/GCN_" + str(fold_idx) + ".txt"
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write("%f %f %f %f" % (avg_loss, acc_train, acc_test, mi_loss))
                    f.write("\n")
            else:
                with open(filename, 'a+') as f:
                    f.write("%f %f %f %f" % (avg_loss, acc_train, acc_test, mi_loss))
                    f.write("\n")
            
            scheduler.step()

            
            torch.cuda.empty_cache()
    print(100*'*')
    print('ASD 10-fold validation results: ')
    print('Model Name: SGSIB')
    print(f"train accuracy list: {acc_train_list}")
    print(f"mean = {acc_train_list.mean()}, variance = {acc_train_list.var()}")
    print(f"test accuracy list: {acc_test_list}")
    print(f"mean = {acc_test_list.mean()}, variance = {acc_test_list.var()}")
    print(100*'*')