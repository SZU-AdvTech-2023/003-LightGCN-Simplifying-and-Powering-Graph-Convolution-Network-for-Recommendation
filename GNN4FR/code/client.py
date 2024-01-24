import torch
from configue import configues
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

class Client():
    def __init__(self,user_id,item_id,dataloader):
        self.item_id = item_id
        self.user_id = user_id
        self.dataloader=dataloader
        self.exclusive_item=None
        self.user_degree=len(self.item_id)
        self.neighbor_users_id = None
        self.neighbor_users_degree = None
        self.item_degree = None
        self.item_neighbor_users_structure = None
        self.hash_index_to_id_of_item=self.get_hash_index_to_id_of_item()
        self.hash_id_to_index_of_item=self.get_hash_id_to_index_of_item()
        self.hash_index_to_id_of_neighbor_users = None
        self.hash_id_to_index_of_neighbor_users = None
        self.adjacency_matrix=None
        self.negative_item_id_for_test=self.get_negative_item_id_for_test()
        self.hash_id_to_index_of_negative_item_for_test=self.get_hash_id_to_index_of_negative_item_for_test()
        self.hash_index_to_id_of_negative_item_for_test=self.get_hash_index_to_id_of_negative_item_for_test()


        self.user_embedding=torch.empty(1,configues["embedding_dim"])
        self.item_embedding=torch.empty(len(self.item_id),configues["embedding_dim"])
        self.neighbor_users_embedding=None
        self.embedding0=None
        self.embedding1=None
        self.embedding2=None
        self.embedding3=None
        self.embedding4=None
        self.embedding=None
        self.negative_item_id=None
        self.hash_id_to_index_of_negative_item=None
        self.hash_index_to_id_of_negative_item=None
        self.negative_item_embedding=None
        self.sample_pairs_for_loss=None
        self.loss=None
        self.embedding_gd=None
        self.negative_item_gd=None
        self.topk_item=None
        self.item_emebding_gd_sent=None

    def get_hash_index_to_id_of_item(self):
        hash_index_to_id_of_item={}
        for index,value in enumerate(self.item_id):
            hash_index_to_id_of_item[index]=value
        return hash_index_to_id_of_item

    def get_hash_id_to_index_of_item(self):
        hash_id_to_index_of_item={}
        for index,value in enumerate(self.item_id):
            hash_id_to_index_of_item[value]=index
        return hash_id_to_index_of_item

    def get_hash_index_to_id_of_neighbor_users(self):
        hash_index_to_id_of_neighbor_users={}
        for index,value in enumerate(self.neighbor_users_id):
            hash_index_to_id_of_neighbor_users[index]=value
        self.hash_index_to_id_of_neighbor_users=hash_index_to_id_of_neighbor_users
        return None

    def get_hash_id_to_index_of_neighbor_users(self):
        hash_id_to_index_of_neighbor_users={}
        for index,value in enumerate(self.neighbor_users_id):
            hash_id_to_index_of_neighbor_users[value]=index
        self.hash_id_to_index_of_neighbor_users=hash_id_to_index_of_neighbor_users
        return None

    def get_hash_id_to_index_of_negative_item(self):
        hash_id_to_index_of_negative_item={}
        for index,value in enumerate(self.negative_item_id):
            hash_id_to_index_of_negative_item[value]=index
        self.hash_id_to_index_of_negative_item=hash_id_to_index_of_negative_item
        return None

    def get_hash_index_to_id_of_negative_item(self):
        hash_index_to_id_of_negative_item={}
        for index,value in enumerate(self.negative_item_id):
            hash_index_to_id_of_negative_item[index]=value
        self.hash_index_to_id_of_negative_item=hash_index_to_id_of_negative_item
        return None

    def get_negative_item_id_for_test(self):
        all_item_id=list(range(self.dataloader.item_number))
        pos_item_id=self.item_id.copy()
        negtive_item_id=list(set(all_item_id)-set(pos_item_id))
        return negtive_item_id

    def get_hash_id_to_index_of_negative_item_for_test(self):
        hash_id_to_index_of_negative_item_for_test={}
        for index,value in enumerate(self.negative_item_id_for_test):
            hash_id_to_index_of_negative_item_for_test[value]=index
        return hash_id_to_index_of_negative_item_for_test

    def get_hash_index_to_id_of_negative_item_for_test(self):
        hash_index_to_id_of_negative_item_for_test={}
        for index,value in enumerate(self.negative_item_id_for_test):
            hash_index_to_id_of_negative_item_for_test[index]=value
        return hash_index_to_id_of_negative_item_for_test


    def get_item_degree(self):
        item_degree={}
        if self.exclusive_item!=None:
            for i in self.exclusive_item:
                item_degree[i]=1
        for i in self.item_neighbor_users_structure.keys():
            item_degree[i]=len(self.item_neighbor_users_structure.get(i))+1
        return item_degree


    # Construct local subgraph
    # Initialize the embedding of layer 0
    def initial_embedding0(self):
        row_length=len(self.neighbor_users_id)+len(self.item_id)+1
        embedding0=torch.nn.Embedding(row_length,configues["embedding_dim"])
        self.neighbor_users_embedding=torch.empty(len(self.neighbor_users_id),configues["embedding_dim"])
        embedding_data=torch.cat([self.user_embedding,self.neighbor_users_embedding],dim=0)
        embedding_data=torch.cat([embedding_data,self.item_embedding],dim=0)
        embedding0.weight.data=embedding_data
        self.embedding0=embedding0
        return None

    # Initialize the embedding of layer 1
    def initial_embedding1(self):
        row_length = len(self.neighbor_users_id) + len(self.item_id) + 1
        embedding1 = torch.nn.Embedding(row_length, configues["embedding_dim"])
        self.embedding1 = embedding1
        return None

    # Initialize the embedding of layer 2
    def initial_embedding2(self):
        row_length = len(self.neighbor_users_id) + len(self.item_id) + 1
        embedding2 = torch.nn.Embedding(row_length, configues["embedding_dim"])
        self.embedding2 = embedding2
        return None

    # Initialize the embedding of layer 3
    def initial_embedding3(self):
        row_length = len(self.neighbor_users_id) + len(self.item_id) + 1
        embedding3 = torch.nn.Embedding(row_length, configues["embedding_dim"])
        self.embedding3 = embedding3
        return None


    # Initialize the embedding of layer 4
    def initial_embedding4(self):
        row_length = len(self.neighbor_users_id) + len(self.item_id) + 1
        embedding4 = torch.nn.Embedding(row_length, configues["embedding_dim"])
        self.embedding4 = embedding4

        return None


    # Initial total embedding
    def initial_embedding(self):
        self.embedding=[self.embedding0,self.embedding1,self.embedding2,self.embedding3,self.embedding4]
        return None


    #Initial the gradients of embedding
    def initial_embedding_gd(self):
        self.embedding_gd=torch.zeros(self.embedding0.weight.shape)
        return None


    def conver_sp_matrix_to_sp_tensor(self,x):
        coo=x.tocoo().astype(np.float32)
        row=torch.Tensor(coo.row).long()
        col=torch.Tensor(coo.col).long()
        index=torch.stack([row,col])
        data=torch.FloatTensor(coo.data)
        sp_tensor=torch.sparse.FloatTensor(index,data,torch.Size(coo.shape))
        return sp_tensor

    # Constructing the adjacency matrix
    def get_adjacency_matrix(self):
        user_array=np.ones((1,len(self.item_id)))
        neighbor_users_array=np.zeros((len(self.neighbor_users_id), len(self.item_id)))
        for i in self.item_neighbor_users_structure.keys():
            users_of_i=self.item_neighbor_users_structure[i]
            i_index=self.hash_id_to_index_of_item[i]
            for u in users_of_i:
                u_index=self.hash_id_to_index_of_neighbor_users[u]
                neighbor_users_array[u_index][i_index]=1.0
        local_ui_interect_matrix=np.concatenate((user_array, neighbor_users_array), axis=0)
        local_user_number=local_ui_interect_matrix.shape[0]
        local_item_number=local_ui_interect_matrix.shape[1]
        local_ui_interect_matrix=csr_matrix(local_ui_interect_matrix)


        adj_matrix = sp.dok_matrix((local_user_number + local_item_number,local_user_number + local_item_number),
                                   dtype=np.float32)
        adj_matrix = adj_matrix.tolil()
        R = local_ui_interect_matrix.tolil()
        adj_matrix[:local_user_number,local_user_number:]=R
        adj_matrix[local_user_number:,:local_user_number]=R.T
        adj_matrix=adj_matrix.todok()

        degree=[float(self.user_degree)]
        for neighbor_u in self.hash_index_to_id_of_neighbor_users.keys():
            id_of_neighbor_user=self.hash_index_to_id_of_neighbor_users[neighbor_u]
            neighbor_u_degree=float(self.neighbor_users_degree[id_of_neighbor_user])
            degree.append(neighbor_u_degree)
        for i in self.hash_index_to_id_of_item.keys():
            id_of_item=self.hash_index_to_id_of_item[i]
            i_degree=float(self.item_degree[id_of_item])
            degree.append(i_degree)

        row_sum = np.array(degree)
        d_inv=np.power(row_sum,-0.5).flatten()
        d_mat=sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_matrix)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        norm_adj_tensor=self.conver_sp_matrix_to_sp_tensor(norm_adj)
        self.adjacency_matrix=norm_adj_tensor

        return None

    def send_k_th_user_embedding(self,k):
        return self.embedding[k].weight.data[0]


    # Synchronize the latest neighbor user embedding received at layer k
    def update_k_th_neighbor_users_embedding(self,k):
        embedding_weight_data=self.embedding[k].weight.data
        for u in range(len(self.neighbor_users_id)):
            neighbor_user_index=u+1
            neighbor_user_id=self.hash_index_to_id_of_neighbor_users[u]
            embedding_weight_data[neighbor_user_index]=self.neighbor_users_embedding[neighbor_user_id]
        return None

    # k-th layer local convolution
    def GNN(self,k):
        embedding_weight_data=self.embedding[k].weight.data
        embedding_weight_data_next_layer = torch.sparse.mm(self.adjacency_matrix, embedding_weight_data)
        self.embedding[k+1].weight.data=embedding_weight_data_next_layer
        return None

    # Sending multi-layer embedding of item i
    def send_item_all_layer_embedding(self,item_id):
        item_all_layer_embedding=[]
        item_index=self.hash_id_to_index_of_item[item_id]
        for i in self.embedding:
            item_embedding=i.weight.data[len(self.neighbor_users_id)+1+item_index]
            item_all_layer_embedding.append(item_embedding)
        return item_all_layer_embedding


    # Initial negative_item_embedding
    def initial_negative_item_embedding(self):
        negative_item_embedding0 = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
        negative_item_embedding1 = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
        negative_item_embedding2 = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
        negative_item_embedding3 = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
        negative_item_embedding4 = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
        self.negative_item_embedding=[negative_item_embedding0,negative_item_embedding1,negative_item_embedding2,
                                      negative_item_embedding3,negative_item_embedding4]
        return None

    # Construction of loss based on sample pairs
    def get_loss(self):
        sum_loss=[]
        for sample_pairs in self.sample_pairs_for_loss:
            pos_item_id=sample_pairs[0]
            neg_item_id=sample_pairs[1]
            pos_item_index=self.hash_id_to_index_of_item[pos_item_id]+len(self.neighbor_users_id)+1
            neg_item_index=self.hash_id_to_index_of_negative_item[neg_item_id]


            final_user_embedding=[]
            for layer in range(configues["layer"]+1):
                final_user_embedding.append(self.embedding[layer].weight[0])

            final_user_embedding=sum(final_user_embedding)/len(self.embedding)

            final_pos_item_embedding=[]
            for layer in range(configues["layer"]+1):
                final_pos_item_embedding.append(self.embedding[layer].weight[pos_item_index])
            final_pos_item_embedding=sum(final_pos_item_embedding)/len(self.embedding)


            final_neg_item_embedding=[]
            for layer in range(configues["layer"]+1):
                final_neg_item_embedding.append(self.negative_item_embedding[layer].weight[neg_item_index])
            final_neg_item_embedding=sum(final_neg_item_embedding)/len(self.embedding)

            pos_scores=torch.mul(final_user_embedding,final_pos_item_embedding)
            pos_scores=torch.sum(pos_scores)

            neg_scores = torch.mul(final_user_embedding, final_neg_item_embedding)
            neg_scores = torch.sum(neg_scores)

            loss=torch.nn.functional.softplus(neg_scores-pos_scores)

            reg_loss=(1/2)*(self.embedding[0].weight[0].norm(2).pow(2)+
                            self.embedding[0].weight[pos_item_index].norm(2).pow(2)+
                            self.negative_item_embedding[0].weight[neg_item_index].norm(2).pow(2))

            loss=loss+configues["weight_decay"]*reg_loss
            sum_loss.append(loss)

        sum_loss=sum(sum_loss)
        self.loss = sum_loss
        return sum_loss


    def loss_backword(self):
        self.loss.backward()
        return None


    def send_k_th_neg_item_gd(self,k):
        neg_item_gd={}
        for index in self.hash_index_to_id_of_negative_item.keys():
            neg_item_id=self.hash_index_to_id_of_negative_item[index]
            neg_item_gd[neg_item_id]=self.negative_item_embedding[k].weight.grad[index]
        return neg_item_gd


    def get_last_layer_embedding_gd(self):
        self.embedding_gd=self.embedding[-1].weight.grad
        return None


    def updata_embedding_gd(self):
        for i in self.negative_item_gd:
            neg_item_id=list(i.keys())[0]
            neg_item_index=self.hash_id_to_index_of_item[neg_item_id]+len(self.neighbor_users_id)+1
            self.embedding_gd[neg_item_index]+=i[neg_item_id]
        return None


    def get_next_layer_gd(self):
        self.embedding_gd=torch.sparse.mm(self.adjacency_matrix, self.embedding_gd)
        return None


    def add_k_th_layer_embedding_gd(self,k):
        if self.embedding[k].weight.grad==None:
            pass
        else:
            self.embedding_gd+=self.embedding[k].weight.grad
        return None



    #Send the embedding_gd of the k-th neighbor user and clear it
    def send_k_th_neighbor_users_gd(self):
        neighbor_users_gd = {}
        for index in range(len(self.neighbor_users_id)):
            embedding_gd=self.embedding_gd.clone()
            neighbor_user_id=self.hash_index_to_id_of_neighbor_users[index]
            neighbor_users_gd[neighbor_user_id]=embedding_gd[index+1]
            self.embedding_gd[index+1]=torch.zeros(configues["embedding_dim"])
        return neighbor_users_gd

    # Divide the gradient to be aggregated into two random parts
    def split_the_item_gradient(self):
        random_item_gd_for_aggregation={}
        for i in self.item_neighbor_users_structure.keys():
            item_index = self.hash_id_to_index_of_item[i] + len(self.neighbor_users_id) + 1
            item_embedding_gd=self.embedding_gd[item_index].clone()
            item_embedding_gd_random=torch.rand(configues["embedding_dim"])
            self.embedding_gd[item_index]=item_embedding_gd-item_embedding_gd_random
            random_item_gd_for_aggregation[i]=item_embedding_gd_random
        return random_item_gd_for_aggregation



    #Sending the gradients of items that require aggregated gradients (except the gradients of exclusive items)
    def send_item_gd_for_aggregation(self):
        item_gd_for_aggregation={}
        for i in self.item_neighbor_users_structure.keys():
            item_index=self.hash_id_to_index_of_item[i]+len(self.neighbor_users_id)+1
            item_gd_for_aggregation[i]=self.embedding_gd[item_index]
        return item_gd_for_aggregation

    #---------------------------------------------------------------------
    #for test

    def send_item_aggregated_embedding(self,item_id):
        item_all_layer_embedding=[]
        item_index=self.hash_id_to_index_of_item[item_id]
        for i in self.embedding:
            item_embedding=i.weight.data[len(self.neighbor_users_id)+1+item_index]
            item_all_layer_embedding.append(item_embedding)
        item_aggregated_embedding=sum(item_all_layer_embedding)/len(self.embedding)
        return item_aggregated_embedding



