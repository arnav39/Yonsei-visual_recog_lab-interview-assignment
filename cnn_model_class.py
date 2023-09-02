import os 
import sys 
import torch
import numpy as np
import pickle
import torch.nn as nn 
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, use_pretrained_embeddings:bool = None, vocab_size = None, embedding_dim = None, n_filters = None,
                 filter_size = None, output_dim = None, n_classes = None):
        
        super().__init__()
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim) if not use_pretrained_embeddings else None 
        self.conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = n_filters, kernel_size = filter_size)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.fc1 = nn.Linear(in_features = n_filters, out_features = output_dim)
        self.fc2 = nn.Linear(in_features = output_dim, out_features = n_classes)
        
    def forward(self, x):
        
        # input_shape = (batch_size, seq_len, embeding_dim) if use pretrained embeddings
        # input_shape = (batch_size, seq_len) if not use pretrained embeddings
        
        if not self.use_pretrained_embeddings:
            x = self.embedding_layer(x)
        
        # x.shape = (batch_size, seq_len, embedding_dim)
        x = F.relu(self.conv1d(x.permute(0, 2, 1))) # conv1d needs dim to be (batch_size, input_channels, seq_len), here input_channels = embedding_dim
        x = self.max_pool(x).squeeze(2)
        # x.shape = (batch_size, n_filters)
        
        # x = F.relu(self.fc1(x))  
        x = self.fc1(x)
        # x.shape = (batch_size, output_dim)
        
        return self.fc2(x)
        # output.shape = (batch_size, n_classes)
        
        
# if __name__ == "__main__":
    
#     device = torch.device("mps:0")
    
#     model = CNN(use_pretrained_embeddings = False,
#                 vocab_size = 10000,
#                 embedding_dim = 100,
#                 n_filters = 100,
#                 filter_size = 3,
#                 output_dim = 100,
#                 n_classes = 2)
    
#     model.to(device)
#     for name, param in model.named_parameters():
#         print(f"name = {name}, device = {param.device}, shape = {param.shape}")
    
    # with open("data/data_train_embeddings/np_train_input_80.pkl", "rb") as f:
    #     input = pickle.load(f)
        
    # input = torch.from_numpy(input).to(device)
    # output = model(input)
    
    # print(f"input.shape = {input.shape}, output.shape = {output.shape}")
    # print(f"input.device = {input.device}, output.device = {output.device}")
    
    # _, pred = torch.max(output, dim=1)
    
    
    
        
        
            