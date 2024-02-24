# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:11:34 2024

@author: Akhil
"""

import numpy as np
import datetime, time
import random
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import Linear, Sequential, GINConv, VGAE, GCNConv
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
import statistics
import gym
import optuna
import optuna.visualization

torch.cuda.empty_cache()
device = torch.device('cpu')

#%% model
class Encoder(nn.Module):
    def __init__(self, hyperparam_dict):
        super(Encoder,self).__init__()
        self.input_dim = 30
        self.hidden_dim1 = hyperparam_dict['self.hidden1']
        self.hidden_dim2 = hyperparam_dict['self.hidden2']
        self.hidden_dim3 = hyperparam_dict['self.hidden3']
        self.hidden_dim4 = hyperparam_dict['self.hidden4']
        self.hidden_dim5 = hyperparam_dict['self.hidden5']
        self.hidden_dim6 = hyperparam_dict['self.hidden6']
        self.hidden_dim7 = hyperparam_dict['self.hidden7']
        self.hidden_dim8 = hyperparam_dict['self.hidden8']
        self.hidden_dim9 = hyperparam_dict['self.hidden9']
        self.output_dim = 9
        self.activation = getattr(nn, hyperparam_dict['self.activation'])
        # self.activation = nn.ReLU()
        self.dropouts = hyperparam_dict['self.droputs']

        self.mlp1 = nn.Sequential(Linear(self.input_dim, self.hidden_dim1,\
                                         weight_initializer=('glorot')),\
                                  self.activation(), Linear(self.hidden_dim1, self.hidden_dim2))
        
        self.mlp2 = nn.Sequential(Linear(self.hidden_dim2, self.hidden_dim3),nn.Dropout(self.dropouts),\
                                  self.activation(),Linear(self.hidden_dim3, self.hidden_dim4))
        
        self.mlp3 = nn.Sequential(Linear(self.input_dim, self.hidden_dim1,\
                                         weight_initializer=('glorot')),\
                                  self.activation(), Linear(self.hidden_dim1, self.hidden_dim2))
                                 
        self.mlp4 = nn.Sequential(Linear(self.hidden_dim2, self.hidden_dim3),nn.Dropout(self.dropouts),\
                                  self.activation(),Linear(self.hidden_dim3, self.hidden_dim4))
            
        self.mlp5 = nn.Sequential(Linear(self.hidden_dim4, self.hidden_dim5),self.activation(),\
                                  Linear(self.hidden_dim5, self.hidden_dim6))
            
        self.mlp6 = nn.Sequential(Linear(self.hidden_dim6, self.hidden_dim7),nn.Dropout(self.dropouts),\
                                  self.activation(),Linear(self.hidden_dim7, self.hidden_dim8))
        
        self.mlp7 = nn.Sequential(Linear(self.hidden_dim8, self.hidden_dim9),self.activation(),\
                                  Linear(self.hidden_dim9, self.output_dim))
        
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)
        self.conv4 = GINConv(self.mlp4)
        self.conv5 = GINConv(self.mlp5)
        self.conv6 = GINConv(self.mlp6)
        self.conv7 = GINConv(self.mlp7)
        
    def forward(self, x, edge_index):
        
        x = x.float()
        z_src = self.conv1(x,edge_index) 
        z_src = self.conv2(z_src, edge_index)
        # z_src = self.conv
        
        z_tar = self.conv3(x, edge_index)
        z_tar = self.conv4(z_tar, edge_index)
        
        return z_src, z_tar
    
class Decoder(nn.Module):
    def __init__(self, hyperparam_dict):
        super(Decoder,self).__init__()
        self.hidden_dim4 = hyperparam_dict['self.hidden4']
        
    def forward(self,z_src, z_tar):
# =============================================================================
#         Asymmetric Innner product decoder. matmul(Z(s),Z(t))
# =============================================================================
#       Inner product decoder--change dimensions in mlp4,mlp2 to hidden_dim4
        
        z_src = z_src.reshape(-1,9,self.hidden_dim4)
        z_tar = z_tar.reshape(-1,9,self.hidden_dim4)
        adj_pred = torch.tensor([])
        for idx, (ele1, ele2) in enumerate(zip(z_src,z_tar)):
            ad = torch.matmul(ele1,ele2.t())
            if len(adj_pred)== 0:
               adj_pred = ad
            else:
                adj_pred = torch.cat((adj_pred,ad), dim = 0)
        
        return adj_pred
    
class GAE(nn.Module):
    def __init__(self, hyperparam_dict):
        super(GAE, self).__init__()
        self.encode = Encoder(hyperparam_dict)
        self.decode = Decoder(hyperparam_dict)
        self.ns_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, edge_index):
        z_src, z_tar = self.encode(x, edge_index)
        adj_pred = self.decode(z_src, z_tar)
        return adj_pred

    def total_loss(self, adj_pred, adj):
        recons_loss = self.ns_loss(adj_pred, adj)
        # kld_loss = -0.5 * torch.mean(torch.sum(1+2*log_var - mu**2 -\
        #                                        log_var.exp()**2, dim = 1))
        # total_loss = recons_loss + kld_loss 
        return recons_loss
#%% dataloading
quick_check = True # switch to quick check. True will switch to quick check. Default is False
print('device is ', device)
total_states = torch.load('states.pt', map_location=('cpu')).to(device)

#%%data processing
states = total_states[:1024]
    
x_o = states[:,:270].reshape(-1,9,30).to(device)
edge_index_o = states[:,391:431].reshape(len(x_o),2,-1).to(torch.long).to(device)
prev_job_o = states[:,270:390].reshape(-1,4,30).to(device)
cur_mach = states[:,390].reshape(len(x_o),1).to(device)
active_edges_o = states[:,431].reshape(len(x_o),1).to(device)

dataset = []

for _x,_edg,_active_edges in\
    zip(x_o,edge_index_o,active_edges_o):
        # generating the adjacency matrix using edge indices
    adj = torch.zeros(9,9).to(device)   
    edges = _edg[:,:int(_active_edges)]
    for i in edges.t():
        adj[i[0]][i[1]] = 1
    dataset.append(Data(x =_x, edge_index = _edg[:,:int(_active_edges)],\
                        adj = adj))

#%% dataset splitting
random.shuffle(dataset)
split = int(len(dataset)*0.7)
train_dataset = dataset[:split]
test_dataset = dataset[split:]
    
#%% optuna hyperparameter tuning

class OptunaOptimizer:
    def __init__(self, epochs: int = 3500, n_trials: int = 50):
        self.epochs = epochs
        
        if quick_check:
            self.n_trials = 1
            
        else:
            self.n_trials = n_trials
            
        self.results = os.path.join(os.getcwd(), 'Results',
                                    f'{datetime.datetime.now().strftime("%d-%m-%Y-%I-%M-%S_%p")}')
        if not os.path.exists(self.results):
            os.makedirs(self.results)
    
    def hyperparameters(self, trial):
        
        hyperparam_dict = {
        'self.batchsize' : trial.suggest_int("Batch_Size", low = 64, high = 128,\
                                             step = 32),
        'self.lr' : trial.suggest_loguniform("Lr", 1e-5, 1e-2),
        'self.optimizer' : trial.suggest_categorical("Optimizer", ["Adam","Adamax",'RMSprop']),
         
        'self.scheduler' : trial.suggest_categorical("Scheduler", ["StepLR"]),
        
        'self.hidden1' : trial.suggest_int("hidden1", low = 40, high = 50),
        'self.hidden2' : trial.suggest_int("hidden2", low = 50, high = 60),
        'self.hidden3' : trial.suggest_int("hidden3", low = 25, high = 40),
        'self.hidden4' : trial.suggest_int("hidden4", low = 11, high = 20),
        'self.hidden5' : trial.suggest_int("hidden5", low = 20, high = 30),
        'self.hidden6' : trial.suggest_int("hidden6", low = 31, high = 38),
        'self.hidden7' : trial.suggest_int("hidden7", low = 39, high = 45),
        'self.hidden8' : trial.suggest_int("hidden8", low = 25, high = 38),
        'self.hidden9' : trial.suggest_int("hidden9", low = 15, high = 24),
        
        'self.activation' : trial.suggest_categorical("activation",
                                                      ["ReLU", "LeakyReLU"]),
        'self.droputs' : trial.suggest_categorical("dropouts", [0.1, 0.2, 0.25, 0.3]),
        'self.weight_decay' : trial.suggest_categorical("weight_decay", [1e-05, 1e-06])
        
       }
        
        return hyperparam_dict
    
    def quickhyperparameters(self, trial):
        quickparamterdict = {'self.batchsize': 96, 'self.lr': 0.0007318402614019153, 'self.optimizer': 'RMSprop', 'self.scheduler': 'StepLR', 'self.hidden1': 44, 'self.hidden2': 60, 'self.hidden3': 38, 'self.hidden4': 19, 'self.hidden5': 22, 'self.hidden6': 35, 'self.hidden7': 44, 'self.hidden8': 29, 'self.hidden9': 22, 'self.activation': 'LeakyReLU', 'self.droputs': 0.1, 'self.weight_decay': 1e-06}
    
        return quickparamterdict
    
    def training(self, trial, batch_size):
        train_loader = DataLoader(train_dataset, batch_size = batch_size,
                                  drop_last = True, shuffle = True)
        ep_train_loss = 0
        train_count = 0
        train_acc = 0 
        train_adj_list = []
        train_link_prob_list = []
        for data in train_loader:
            train_count+=1
            x, edge_index,train_adj = data.x, data.edge_index, data.adj
            z_src, z_tar = self.model.encode(x, edge_index)
            train_adj_pred = self.model.decode(z_src, z_tar)
            train_loss = self.model.total_loss(train_adj_pred, train_adj)
            ep_train_loss += train_loss.item()
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
# =============================================================================
#                     get gradients and weights for each step
# =============================================================================
#                 for name, params in model.named_parameters():
#                     print('gradients', params.grad.mean())
#                     print('weights', params.data)
# =============================================================================
#          prediciting the probabaility of the link between the nodes
# =============================================================================
            train_rounded_adj_pred = torch.sigmoid(train_adj_pred).round().detach()
            train_link = torch.mul(train_rounded_adj_pred, train_adj)
            train_link_prob = ((train_link.sum())/(train_adj.sum())).cpu().item()
            train_link_prob_list.append(train_link_prob)
            train_adj_list.append(train_rounded_adj_pred)
            
        return train_adj_list, train_link_prob_list, ep_train_loss
        
    def validation(self, trial, batch_size):
        test_loader = DataLoader(test_dataset, batch_size = batch_size,
                                 drop_last = True, shuffle = True)
        with torch.no_grad():
            ep_val_loss = 0
            val_count = 0
            val_acc = 0 
            val_adj_list = []
            val_link_prob_list = []
            for data in test_loader:
                val_count+=1
                x, edge_index,val_adj = data.x, data.edge_index, data.adj
                z_src,z_tar = self.model.encode(x, edge_index)
                val_adj_pred = self.model.decode(z_src, z_tar)
                val_loss = self.model.total_loss(val_adj_pred, val_adj)
                ep_val_loss += val_loss.item()
                val_rounded_adj_pred = torch.sigmoid(val_adj_pred).round().detach()
                val_link = torch.mul(val_rounded_adj_pred, val_adj)
                val_link_prob = ((val_link.sum())/(val_adj.sum())).cpu().item()
                val_link_prob_list.append(val_link_prob)
                val_adj_list.append(val_rounded_adj_pred)
                
        return val_adj_list, val_link_prob_list, ep_val_loss
    
    def train_val(self, trial):
        t = time.time()
        if quick_check:
            hyperparam_dict = self.quickhyperparameters(trial)
            
        else:
            hyperparam_dict = self.hyperparameters(trial)
        
        self.model = GAE(hyperparam_dict).to(device)
        self.model.train()
        
        self.lr = hyperparam_dict["self.lr"]
        self.weight_decay = hyperparam_dict["self.weight_decay"]
        self.optimizer = getattr(optim, hyperparam_dict["self.optimizer"])(self.model.parameters(),
                                                                           lr = self.lr,
                                                                           weight_decay = self.weight_decay)
        self.batch_size = hyperparam_dict["self.batchsize"]
        
        torch.autograd.set_detect_anomaly(True)
        epochs = self.epochs
        tot_train_loss = []        
        train_link_pred_acc = []
        train_tot_adj_pred = []
        
        tot_val_loss = []        
        val_link_pred_acc = []
        val_tot_adj_pred = []
        
        for ep in range(epochs):
            self.model.train()
            train_adj_list, train_link_prob_list, ep_train_loss = self.training(trial, self.batch_size)
            self.model.eval()
            val_adj_list, val_link_prob_list, ep_val_loss = self.validation(trial, self.batch_size)
                
            train_tot_adj_pred.append(train_adj_list)
            train_ep_link_prob = np.mean(train_link_prob_list)
            train_link_pred_acc.append(train_ep_link_prob)
            train_link_pred_accuracy = np.max(train_link_pred_acc)
            tot_train_loss.append(ep_train_loss)
            
            val_tot_adj_pred.append(val_adj_list)
            val_ep_link_prob = np.mean(val_link_prob_list)
            val_link_pred_acc.append(val_ep_link_prob)
            val_link_pred_accuracy = np.max(val_link_pred_acc)
            tot_val_loss.append(ep_val_loss)
            
            if ep % 500 == 0:
                print("Epoch:", '%04d' % (ep), "train_loss=", "{:.5f}".format(ep_train_loss),\
                        "train_link_pred=", "{:.5f}".format(train_ep_link_prob),"train_time=", "{:.5f}".format(time.time() - t))
        
                print("Epoch:", '%04d' % (ep), "val_loss=", "{:.5f}".format(ep_val_loss),\
                        "val_link_pred=", "{:.5f}".format(val_ep_link_prob),"val_time=", "{:.5f}".format(time.time() - t))
                    
        self.trial_loc = os.path.join(self.results, f'trial_{trial.number}')
        if not os.path.exists(self.trial_loc):
            os.makedirs(self.trial_loc)
        
        with open(os.path.join(self.trial_loc, 'hyperparmeters.txt'), 'w' ) as outfile:
            outfile.write(str(hyperparam_dict))
            
        plt.title('train_loss')
        plt.plot(tot_train_loss)
        plt.grid()
        plt.savefig(os.path.join(self.trial_loc,'train_loss.jpg'))
        plt.show()
        
        plt.title('link prediction accuracy training')
        plt.plot(train_link_pred_acc)
        plt.grid()
        plt.savefig(os.path.join(self.trial_loc,'train_link_prob.jpg'))
        plt.show()
        
        plt.title('validation loss')
        plt.plot(tot_val_loss)
        plt.grid()
        plt.savefig(os.path.join(self.trial_loc,'validation_loss.jpg'))
        plt.show()
        
        plt.title('link prediction accuracy validation')
        plt.plot(val_link_pred_acc)
        plt.grid()
        plt.savefig(os.path.join(self.trial_loc,'validation_link_probability.jpg'))
        plt.show()
        
        return train_link_pred_accuracy, train_tot_adj_pred, self.model.encode
        
# =============================================================================
#     def testing (self):
#         self.model.eval()
#         test_loader = DataLoader(test_dataset, batch_size = self.batch_size,\
#                                   drop_last = True,
#                                   shuffle = True)
#         test_link_prob = []
#         for data in test_loader:
#             x, edge_index, adj = data.x, data.edge_index, data.adj
#             with torch.no_grad():
#                 z_src = self.model.encode1(x, edge_index)
#                 z_tar = self.model.encode2(x, edge_index)
#                 adj_pred = self.model.decode(z_src, z_tar)
#             rounded_adj = torch.sigmoid(adj_pred).round().detach()
#             link = torch.mul(rounded_adj, adj)
#             link_prob = ((link.sum())/(adj.sum())).cpu().item()        
#             test_link_prob.append(link_prob)
#             test_lin_prob = np.array(test_link_prob)
#                          
#         final_accuracy = test_lin_prob.mean()
#         print('final_accuracy is ', final_accuracy)
#         plt.title('test_acc')
#         plt.plot(test_link_prob)
#         plt.savefig(os.path.join(self.trial_loc,'test_acc.jpg'))
#         plt.show()
# =============================================================================
        
    def objective(self, trial):
        torch.cuda.empty_cache()
        pred_acc, adj_pred, enc_model = self.train_val(trial)
        # self.testing()
        torch.save(pred_acc, os.path.join(self.trial_loc,'pred_acc.pt'))
        torch.save(adj_pred, os.path.join(self.trial_loc,'adj_pred.pt'))
# =============================================================================
#         with open(os.path.join(self.trial_loc,'pred_acc.txt'), 'w') as outfile:
#             json.dump(pred_acc, outfile)
#         with open(os.path.join(self.trial_loc,'adj_pred.txt'), 'w') as outfile:
#             json.dump(adj_pred, outfile)
#             
# =============================================================================
        if trial.number==0:
            with open(os.path.join(self.results,'best_pred_acc.txt'), 'w') as outfile:
                json.dump([trial.number, pred_acc], outfile)
            print('Saving best trial')
            # torch.save(model.state_dict(), save_location)
            
        else:
           with open(os.path.join(self.results,'best_pred_acc.txt'), 'rb') as outfile:
               best_val = json.load(outfile)[1]
           # with open('best_pred_acc.txt', 'rb') as outfile:
           #     best_val = json.load(outfile)
           print('saving best accuracy')

           if pred_acc > best_val:
               with open(os.path.join(self.results,'best_pred_acc.txt'), 'w') as outfile:
                   json.dump([trial.number, pred_acc], outfile)
                   
               print('Saving best trial')
               # torch.save(model.state_dict(), save_location)
                   
        return pred_acc

    # def optimize(self, n_trials):
        # self.study = optuna.create_study(direction = 'maximize')
        # self.study.optimize(self.objective, n_trials)
    
    def start_study(self):  
        study = optuna.create_study(direction = 'maximize')
        study.optimize(self.objective, n_trials = self.n_trials)
        
    def savemodel(self):
        torch.save(self.model.encode.state_dict(),"encoder_model.pt")
        return self.model.encode
#%%

# optuna_study = OptunaOptimizer.optimize(n_trials=20)

optimizing = OptunaOptimizer()
optimizing.start_study()
encoded_model = optimizing.savemodel()

