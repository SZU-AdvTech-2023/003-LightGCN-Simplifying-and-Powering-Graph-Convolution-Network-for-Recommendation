
import numpy as np
class Sample():
    def __init__(self,dataloader):
        self.all_pos_item=dataloader.all_pos_items
        self.train_data_num=dataloader.train_data_num
        self.user_number=dataloader.user_number
        self.item_number=dataloader.item_number
    def sample_pairs(self,epoch):
        '''

        :return: (user_id,pos_item_id,neg_item_id),2d_array
        '''
        sample_pairs=[]
        np.random.seed(epoch)
        users_sample_id=np.random.randint(0,self.user_number,5*self.train_data_num)
        for index,user_id in enumerate(users_sample_id):
            pos_item_of_user=self.all_pos_item[user_id]
            if len(pos_item_of_user)==0:
                continue
            pos_index=np.random.randint(0,len(pos_item_of_user))
            pos_item_id=pos_item_of_user[pos_index]
            while True:
                neg_item_id=np.random.randint(0,self.item_number)
                if neg_item_id in pos_item_of_user:
                    continue
                else:
                    break
            sample_pairs.append([user_id,pos_item_id,neg_item_id])
        sample_pairs=np.array(sample_pairs)
        return sample_pairs
