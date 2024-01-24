from dataloader import Dataloader
import numpy as np
from configue import configues
import torch
from client import Client
from server import Server
import utils
from torch import optim



def test():
    """
    test function
    :return:
    """
    #Server collects all item_embedding
    all_item_embedding_for_test = server.get_all_item_embedding_for_test()

    #All users get top-k recommended items
    for client in clients:
        item_embedding=all_item_embedding_for_test[client.hash_index_to_id_of_negative_item_for_test[0]]
        item_embedding=torch.unsqueeze(item_embedding, 0)
        for i in range(len(client.negative_item_id_for_test)-1):
            item_embedding_mid=all_item_embedding_for_test[client.hash_index_to_id_of_negative_item_for_test[i+1]]
            item_embedding_mid=torch.unsqueeze(item_embedding_mid, 0)
            item_embedding=torch.cat((item_embedding,item_embedding_mid), dim=0)
        final_user_embedding = []
        for layer in range(configues["layer"] + 1):
            final_user_embedding.append(client.embedding[layer].weight[0])
        final_user_embedding = sum(final_user_embedding) / len(client.embedding)

        activation_function=torch.nn.Sigmoid()
        rating=activation_function(torch.matmul(final_user_embedding,item_embedding.t()))

        index = torch.topk(rating, configues["topk"])
        top_k=np.array(index[1])
        top_k_item_id=[client.hash_index_to_id_of_negative_item_for_test[element] for element in top_k]

        client.topk_item=top_k_item_id

    #print precision and recall
    precision, recall = server.get_precision()
    print("precision:"+str(precision))
    print("recall:"+str(recall))
    return None


#start

#load data
dataloader=Dataloader()


#initial user_embedding and item_embedding
torch.manual_seed(configues["seed"])
users_embedding_weight = torch.nn.Embedding(num_embeddings=dataloader.user_number, embedding_dim=configues["embedding_dim"]).weight
items_embedding_weight = torch.nn.Embedding(num_embeddings=dataloader.item_number, embedding_dim=configues["embedding_dim"]).weight


#initial clients
clients=[]
for u in range(dataloader.user_number):
    item_id=list(dataloader.train_user_item_matrix[u].nonzero()[1])
    client=Client(u,item_id,dataloader)
    user_embedding=users_embedding_weight.data[u]
    client.user_embedding[0]=user_embedding

    for i in item_id:
        item_embedding=items_embedding_weight.data[i]
        client.item_embedding[client.hash_id_to_index_of_item[i]]=item_embedding
    clients.append(client)


#initial the server
server=Server(clients,dataloader)

#server distribute neighbor users
# (sent neighbor users id,exclusive item id,neighbor users embedding and degree to the corresponding client)
neighbor_users_of_all_clients,exclusive_items_of_user,clients_degree=server.distribute_neighbor_user()
neighbors_users_id_of_all_clients=server.get_neighbors_users_id_of_all_clients(neighbor_users_of_all_clients)


#client receive the infomation and expand the local graph
for index,client in enumerate(clients):
    client.neighbor_users_id=neighbors_users_id_of_all_clients[index]
    client.get_hash_index_to_id_of_neighbor_users()
    client.get_hash_id_to_index_of_neighbor_users()
    client.exclusive_item=exclusive_items_of_user.get(index)
    client.item_neighbor_users_structure=neighbor_users_of_all_clients.get(index)
    neighbor_users_degree={}
    for i in client.neighbor_users_id:
        neighbor_users_degree[i] = clients_degree[i]
    client.neighbor_users_degree = neighbor_users_degree
    client.item_degree=client.get_item_degree()


#Initialize the embedding of each layer and construct the adjacency matrix
for client in clients:
    #Initialize the embedding of each layer
    client.initial_embedding0()
    client.initial_embedding1()
    client.initial_embedding2()
    client.initial_embedding3()
    client.initial_embedding4()
    client.initial_embedding()
    client.initial_embedding_gd()
    #construct the adjacency matrix
    client.get_adjacency_matrix()

#Get learnable embedding
embedding_study=[]
for client in clients:
    embedding_study.append(client.embedding0.weight)
#Initialize Optimizer
opt=optim.Adam(embedding_study,lr=configues["lr"])


#start to train

for epoch in range(configues["epochs"]):

    # The server samples the trained sample pairs
    batch_size = configues["train_batch"]
    S = server.sample_pairs(epoch)

    #The server disrupts the sampled samples and divides them into several batches for training
    users=S[:,0]
    pos_items = S[:,1]
    neg_items = S[:,2]

    users,pos_items,neg_items=utils.shuffle(users,pos_items,neg_items)
    iter_time=len(users)//batch_size+1
    aver_loss=0.

    #set a flag for printing train loss during training process
    flag=1
    for (batch_users,batch_pos_items,batch_neg_items) in utils.minibatch(users,
                                                                         pos_items,
                                                                         neg_items,
                                                                         batch_size=configues["train_batch"]):
        #forward propagation
        opt.zero_grad()
        # sync neighbor_user_embedding
        for layer in range(configues["layer"]):
            k_th_users_embedding_of_all_clients = server.distribute_k_th_users_embedding(layer)
            for client in clients:
                client.neighbor_users_embedding = k_th_users_embedding_of_all_clients[client.user_id]
                client.update_k_th_neighbor_users_embedding(layer)
                client.GNN(layer)

        # The server collects the embedding of the negative samples used in the current batch
        all_item_embedding_needed = server.get_all_item_embedding(batch_neg_items)  # return format:{item_id:all_layer_embedding_of_item}

        negative_item_id_of_sample_clients=server.get_negative_item_id_of_sample_clients(batch_users,batch_neg_items)

        # Sampled users receive the negative_item_id and the corresponding embedding
        for client_id in negative_item_id_of_sample_clients.keys():
            #receive the negative_item_id
            clients[client_id].negative_item_id = negative_item_id_of_sample_clients[client_id]

            #receive the corresponding embedding
            clients[client_id].get_hash_id_to_index_of_negative_item()
            clients[client_id].get_hash_index_to_id_of_negative_item()
            clients[client_id].initial_negative_item_embedding()
            for index in range(len(clients[client_id].negative_item_id)):
                negative_item_id=clients[client_id].hash_index_to_id_of_negative_item[index]
                for i,value in enumerate(clients[client_id].negative_item_embedding):
                    value.weight.data[index]=all_item_embedding_needed[negative_item_id][i]


        #The server assigns each sampled user a sample pair to build a loss
        sample_pairs_for_loss_of_sample_clients=server.get_sample_pairs_for_loss(batch_users,batch_pos_items,batch_neg_items)

        sum_loss=0.0
        for client_id in negative_item_id_of_sample_clients.keys():
            clients[client_id].sample_pairs_for_loss = sample_pairs_for_loss_of_sample_clients[client_id]
            loss_of_one_client=clients[client_id].get_loss()
            sum_loss+=loss_of_one_client
        aver_loss+=sum_loss

        print(sum_loss.item())

        if flag:
            test()
            flag=0


        #backward propagation
        for client_id in negative_item_id_of_sample_clients.keys():
            clients[client_id].loss_backword()

        #Send the gradient of negative samples back to the corresponding client
        negative_item_gd_of_clients=server.get_negative_item_gd(batch_neg_items,configues["layer"])

        #User receives negative samples of the gradient
        for client_id in negative_item_gd_of_clients.keys():
            clients[client_id].negative_item_gd=negative_item_gd_of_clients[client_id]


        #The user participating in the training takes out the last layer of grad
        for client_id in negative_item_id_of_sample_clients.keys():
            clients[client_id].get_last_layer_embedding_gd()

        #The user receiving the negative sample updates the gradient of the negative sample
        for client_id in negative_item_gd_of_clients.keys():
            clients[client_id].updata_embedding_gd()



        #All users solve for the gradient of the next layer until they reach layer 0
        for layer in range(configues["layer"]):
            for index,client in enumerate(clients):

                client.get_next_layer_gd()
                client.add_k_th_layer_embedding_gd(3-layer)

            # Gradient for back propagation of negative samples
            negative_item_gd_of_clients = server.get_negative_item_gd(batch_neg_items, 3-layer)
            # User receives negative samples of the gradient
            for client_id in negative_item_gd_of_clients.keys():
                clients[client_id].negative_item_gd = negative_item_gd_of_clients[client_id]
            # The user receiving the negative sample updates the gradient of the negative sample
            for client_id in negative_item_gd_of_clients.keys():
                clients[client_id].updata_embedding_gd()
            # Pass back the gradient of the neighbor user and zero out the neighbor embedding_gd
            neighbor_users_embedding_gd=server.get_k_th_neighbor_users_embedding_gd()
            for client_id in neighbor_users_embedding_gd.keys():
                clients[client_id].embedding_gd[0]+=neighbor_users_embedding_gd[client_id]

        # The server forwards a randomly uploaded gradient for each user
        item_embedding_gd_of_all_clients=server.transmit_item_embedding_gd()    #return format:{user_id:[{item_id:item_embedding_gd]}
        #Each user receives the corresponding item_embedding_gd, and adds it to their own gradient
        for client in clients:
            random_item_gd=item_embedding_gd_of_all_clients[client.user_id]
            if random_item_gd==[]:
                continue
            for i in random_item_gd:
                item_id=list(i.keys())[0]
                item_index = client.hash_id_to_index_of_item[item_id] + len(client.neighbor_users_id) + 1
                client.embedding_gd[item_index] += i[item_id]

        # The server aggregate the gradients of items
        item_gd_after_aggregation=server.aggregate_item_embedding_gd()
        # All users receive the gradient after item aggregation
        for client in clients:
            for item_id in client.item_neighbor_users_structure.keys():
                item_index=client.hash_id_to_index_of_item[item_id]+len(client.neighbor_users_id)+1
                client.embedding_gd[item_index]=item_gd_after_aggregation[item_id]

        # Write the obtained layer 0 embedding_gd into the original embedding
        for client in clients:
            client.embedding0.weight.grad=client.embedding_gd

        opt.step()
        #Set the gradient of all layers except layer 0 to zero
        #Zero out the embedding_gd of all clients
        for client in clients:
            client.embedding1.weight.grad = None
            client.embedding2.weight.grad = None
            client.embedding3.weight.grad = None
            client.embedding4.weight.grad = None
            client.initial_embedding_gd()
    aver_loss = aver_loss / iter_time
    print("epoch-" + str(epoch) + ":" + str(aver_loss))



