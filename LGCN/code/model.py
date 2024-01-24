from torch import nn
from configue import configues
import torch
from Dataloader import Dataloader
class Light_GCN(nn.Module):
    def __init__(self,dataloader,configues=configues):
        super(Light_GCN,self).__init__()
        self.configues=configues
        self.dataloader=dataloader
        self.user_number=self.dataloader.user_number
        self.item_number=self.dataloader.item_number
        self.embedding_dim=self.configues["embedding_dim"]
        self.layer=self.configues["layer"]
        self.keep_prob=self.configues["keep_prob"]
        self.weight_decay=self.configues["weight_decay"]
        self.training=True

        torch.manual_seed(2023)
        self.users_embedding=torch.nn.Embedding(num_embeddings=self.user_number,embedding_dim=self.embedding_dim)
        self.items_embedding=torch.nn.Embedding(num_embeddings=self.item_number,embedding_dim=self.embedding_dim)

        self.f=nn.Sigmoid()
        self.adjacency_matrix=dataloader.adjacency_matrix


    def forward(self):
        """
        :return: The embedding of the user and item used for the final prediction
        """
        user_embedding=self.users_embedding.weight
        item_embedding=self.items_embedding.weight
        all_embedding=torch.cat([user_embedding,item_embedding])
        embeddings=[all_embedding]

        if self.training:
            adjacency_matrix_drop_out=self.drop_out(self.adjacency_matrix,self.keep_prob)
        else:
            adjacency_matrix_drop_out=self.adjacency_matrix

        for layer in range(self.layer):
            all_embedding=torch.sparse.mm(adjacency_matrix_drop_out,all_embedding)
            embeddings.append(all_embedding)
        embeddings = torch.stack(embeddings, dim=1)
        final_embedding=torch.mean(embeddings,dim=1)
        final_users_embeddings,final_items_embeddings=torch.split(final_embedding,[self.user_number,self.item_number])
        return final_users_embeddings,final_items_embeddings

    def get_all_kinds_embedding(self,users_id,poses_item_id,negs_item_id):
        final_users_embeddings, final_items_embeddings=self.forward()
        users_embedding=final_users_embeddings[users_id]
        poses_item_embedding=final_items_embeddings[poses_item_id]
        negs_item_embedding=final_items_embeddings[negs_item_id]
        users_embedding0=self.users_embedding(users_id)
        poses_item_embedding0=self.items_embedding(poses_item_id)
        negs_item_embedding0=self.items_embedding(negs_item_id)
        return users_embedding,poses_item_embedding,negs_item_embedding,users_embedding0,poses_item_embedding0,negs_item_embedding0
    def get_loss(self,users_id,poses_item_id,negs_item_id):
        (users_embedding, poses_item_embedding, negs_item_embedding,users_embedding0, poses_item_embedding0,
         negs_item_embedding0)=self.get_all_kinds_embedding(users_id.long(),poses_item_id.long(),negs_item_id.long())


        reg_loss=(1/2)*(users_embedding0.norm(2).pow(2)+
                        poses_item_embedding0.norm(2).pow(2)+
                        negs_item_embedding0.norm(2).pow(2))



        pos_scores=torch.mul(users_embedding,poses_item_embedding)
        pos_scores=torch.sum(pos_scores,dim=1)
        neg_scores=torch.mul(users_embedding,negs_item_embedding)
        neg_scores=torch.sum(neg_scores,dim=1)
        loss=torch.sum(torch.nn.functional.softplus(neg_scores-pos_scores))

        sum_loss=loss+self.weight_decay*reg_loss

        print(sum_loss.item())

        return sum_loss


    def drop_out(self,x,keep_prob):
        size=x.size()
        index=x.indices().t()
        values=x.values()
        random_index=torch.rand(len(values))+keep_prob
        random_index=random_index.int().bool()
        index=index[random_index]
        values=values[random_index]/keep_prob
        result=torch.sparse.FloatTensor(index.t(),values,size)
        return result

    def get_users_rating(self,users):
        final_users_embeddings, final_items_embeddings = self.forward()
        users_embedding=final_users_embeddings[users.long()]
        items_embedding=final_items_embeddings
        rating=self.f(torch.matmul(users_embedding,items_embedding.t()))
        return rating

    def get_one_user_rating(self,user):
        '''
        :param user: user_id
        :return: rating_of_all_items
        '''
        final_users_embeddings, final_items_embeddings = self.forward()
        user_embedding=final_users_embeddings[user.long()]
        items_embedding = final_items_embeddings
        rating = self.f(torch.matmul(user_embedding, items_embedding.t()))
        return rating

