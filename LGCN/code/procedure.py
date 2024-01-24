import torch
import utils
import evalution

from configue import configues
def train(sampeler,dataloader,model,configues,opt,epoch):
    recmodel=model
    recmodel.train()
    recmodel.training=True
    batch_size=configues["train_batch"]
    S=sampeler.sample_pairs(epoch)
    users=torch.Tensor(S[:,0]).long()
    pos_items=torch.Tensor(S[:,1]).long()
    neg_items=torch.Tensor(S[:,2]).long()

    users=users.to(configues["device"])
    pos_items=pos_items.to(configues["device"])
    neg_items=neg_items.to(configues["device"])


    users,pos_items,neg_items=utils.shuffle(users,pos_items,neg_items)
    iter_time=len(users)//batch_size+1
    aver_loss=0.

    for (batch_users,batch_pos_items,batch_neg_items) in utils.minibatch(users,
                                                                         pos_items,
                                                                         neg_items,
                                                                         batch_size=configues["train_batch"]):

        loss=model.get_loss(batch_users,batch_pos_items,batch_neg_items)
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss+=loss.cpu().item()
    aver_loss=aver_loss/iter_time
    return aver_loss

def test(model,configues,dataloader):
    recmodel=model
    evalutioner=evalution.Evalution(model=model,dataloader=dataloader,configues=configues)
    model.training=False
    recmodel.eval()
    with torch.no_grad():
        precision=evalutioner.get_precision_of_all_u()
        recall=evalutioner.get_recall_of_all_u()
        return precision,recall