from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
import pickle
import argparse
import math
from matplotlib import pyplot as plt



def accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score(pred, target, num_classes):
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score






class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.weight1 = nn.Parameter(torch.Tensor(self.w_out*self.num_channels, w_out))
        self.weight2 = nn.Parameter(torch.Tensor(self.w_out*self.num_channels, w_in))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.lossR = loss = nn.BCELoss()
        self.lossC = nn.CrossEntropyLoss()
        # self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def gcn_conv1(self,X,H):
        X = torch.mm(X, self.weight1)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def gcn_conv2(self,X,H):
        X = torch.mm(X, self.weight2)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        AA = A.permute(2,0,1)
        A_=AA[0]
        for i in range(1,self.num_edge):
            A_ = A_ + AA[i]
            
            
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)
        for i in range(self.num_channels):
            if i==0:
                H_ = H[i]
            else:
                H_ = H_+H[i]
        X_r = self.gcn_conv2(X_,H_)        
        X_ = self.gcn_conv1(X_,H_)
        y = self.linear2(X_[target_x])
        X_r = torch.where(X_r > 1, torch.sigmoid(X_r), X_r)
        X_r = torch.where(X_r < 0, torch.sigmoid(X_r), X_r)

        lossR = self.lossR(X_r,X)
        loss = self.lossC(y, target)+lossR
        return loss, y, Ws, lossR

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='ACM',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='true',
                        help='adaptive learning rate')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
        print(edges)
    with open('data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]

    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)
    
    num_classes = torch.max(train_target).item() + 1
    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params':model.weight},
                                        {'params':model.weight1},
                                        {'params':model.weight2},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        # loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1)
            model.zero_grad()
            model.train()
            loss,y_train,Ws,lossR = model(A, node_features, train_node, train_target)
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {},lossR: {}'.format(loss.detach().cpu().numpy(), train_f1,lossR.detach().cpu().numpy()))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_,_ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,_,_ = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1 
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1
