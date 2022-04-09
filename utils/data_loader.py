import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch

class Loader_list(Dataset): 
    def __init__(self, lst):
        self.lst = lst
    
    def __getitem__(self, index):
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

class Block:
    def __init__(self, mat, u_batch_size = 1000, i_batch_size = 1000, device = 'cuda'): 
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size, shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size, shuffle=True, num_workers=0)
    
    def get_batch(self, batch_user, batch_item, device = 'cuda'): 
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())
        index = torch.tensor(np.where(index_row * index_col)[0]).to(device)

        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]

class NoLabelBlock:
    def __init__(self, mat, times, device = 'cuda'): 
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.TrainSize = self.mat._nnz()
        index_row = np.random.choice(np.arange(self.n_users), size= self.TrainSize* times *2, replace=True)
        index_col = np.random.choice(np.arange(self.n_items), size= self.TrainSize* times *2, replace=True)
        Pool_u = []
        Pool_i = []
        for i in range(len(index_row)):
            if len(Pool_u) >= self.TrainSize * times:
                break
            else:
                if self.mat[index_row[i], index_col[i]] == 0:
                    Pool_u.append(index_row[i])
                    Pool_i.append(index_col[i])
        for i in range(len(index_row)):
            if len(Pool_u) >= self.TrainSize * times:
                break
            else:
                if self.mat[index_row[i], index_col[i]] == 0:
                    Pool_u.append(index_row[i])
                    Pool_i.append(index_col[i])
        self.u_no = torch.tensor(Pool_u)
        self.i_no = torch.tensor(Pool_i)
    def get_nolabel_batch(self, batch_user, batch_item, device = 'cuda'): 
        index_ba_u = np.isin(self.u_no.numpy(), batch_user.cpu().numpy())
        index_ba_i = np.isin(self.i_no.numpy(), batch_item.cpu().numpy())
        index = torch.tensor(np.where(index_ba_u * index_ba_i)[0]).to(device)

        return self.u_no[index].to(device), self.i_no[index].to(device)


    def get_batch_Wneg(self, batch_user, batch_item, neg_item_all, neg_users, device = 'cuda'):
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())
        index = torch.tensor(np.where(index_row * index_col)[0]).to(device)
        neg_item=torch.tensor(neg_item_all)[index[~torch.gt(index, len(neg_item_all)-1)]].to(device)
        neg_user=torch.tensor(neg_users)[index[~torch.gt(index, len(neg_users)-1)]].to(device)

        all_users = torch.cat( [self.mat._indices()[0][index], neg_user] )
        all_items = torch.cat( [self.mat._indices()[1][index], neg_item] )
        all_values = torch.cat( [self.mat._values()[index], -1*torch.ones(len(neg_item)).to(device)] )

        return all_users, all_items, all_values
 
class User: 
    def __init__(self, mat_position, mat_rating, u_batch_size = 100, device = 'cuda'):
        self.mat_position = mat_position
        self.mat_rating = mat_rating
        self.n_users, self.n_items = self.mat_position.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size, shuffle=True, num_workers=0)
    
    def get_batch(self, batch_user, device = 'cuda'): 
        index = np.isin(self.mat_position._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index = torch.tensor(np.where(index)[0]).to(device)
        # index = (self.mat_pisition._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index = torch.where(index_row)[0]
        return self.mat_position._indices()[0][index], self.mat_position._indices()[1][index], self.mat_position._values()[index], self.mat_rating._values()[index]

class Interactions(Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape

    def __getitem__(self, index):
        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        # return torch.tensor(int(row)), torch.tensor(int(col)), torch.tensor(val)
        return row, col, val

    def __len__(self):
        return self.mat._nnz()