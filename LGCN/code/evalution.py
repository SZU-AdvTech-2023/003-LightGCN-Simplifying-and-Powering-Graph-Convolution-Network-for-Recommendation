import torch
import numpy as np

class Evalution():
    def __init__(self,model,dataloader,configues):
        self.model=model
        self.dataloader=dataloader
        self.configues=configues
        self.k=self.configues["topk"]
    def get_top_k(self):
        '''

        :return: top_k items of all users
        '''
        top_k={}
        for user_id in range(self.dataloader.user_number):
            rec_items=[]
            user_id_longtensor=torch.LongTensor([user_id])
            rating=self.model.get_one_user_rating(user_id_longtensor).cpu()
            sort_index=torch.topk(rating,self.dataloader.item_number)
            rating_index=np.array(sort_index[1]).squeeze()
            pos_item=self.dataloader.all_pos_items[user_id]
            for item in rating_index:
                if item in pos_item:
                    continue
                else:
                    rec_items.append(item)
                if len(rec_items)==self.k:
                    break
            top_k[user_id]=rec_items
        return top_k

    def get_sorted_rating(self):
        '''
        :return: sorted items of all users according rating
        dict{user_id:sorted items}
        '''
        sorted_rating={}
        for user_id in range(self.dataloader.user_number):
            rec_items=[]
            user_id_longtensor=torch.LongTensor([user_id])
            rating=self.model.get_one_user_rating(user_id_longtensor)
            sort_index=torch.topk(rating,self.dataloader.item_number)
            rating_index=np.array(sort_index[1]).squeeze()
            sorted_rating[user_id]=list(rating_index)
        return sorted_rating


    def get_precision_of_all_u(self):
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        topk_of_rec_list_of_all_u=self.get_top_k()
        precision_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            common_item = list(set(topk_of_rec_list_of_u) & set(true_list))
            precision_of_all_u_sum += len(common_item) / self.k
        precision_of_all_u = precision_of_all_u_sum / len(haxi_of_u_to_i_test.keys())
        return precision_of_all_u

    def get_recall_of_all_u(self):
        topk_of_rec_list_of_all_u = self.get_top_k()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        recall_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:

            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            common_item = list(set(topk_of_rec_list_of_u) & set(true_list))
            recall_of_all_u_sum += len(common_item) / len(true_list)
        recall_of_all_u = recall_of_all_u_sum / len(haxi_of_u_to_i_test.keys())
        return recall_of_all_u




