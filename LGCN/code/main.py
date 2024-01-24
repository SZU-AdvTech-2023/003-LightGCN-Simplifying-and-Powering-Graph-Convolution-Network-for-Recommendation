from Dataloader import Dataloader
from sample import Sample
from model import Light_GCN
from configue import configues
from torch import optim
import procedure
import time
from torch.utils.tensorboard import SummaryWriter

dataloader=Dataloader()
sampeler=Sample(dataloader)
model=Light_GCN(dataloader,configues=configues)
model=model.to(configues["device"])
opt=optim.Adam(model.parameters(),lr=configues["lr"])
tb_writer=SummaryWriter(log_dir="../experiment/tensorboard")


for epoch in range(configues["epochs"]):

    precision,recall=procedure.test(model,configues,dataloader)
    print("precision:"+str(precision))
    print("recall:"+str(recall))
    tb_writer.add_scalar("precision",precision,epoch)
    tb_writer.add_scalar("recall", recall, epoch)
    start=time.time()
    aver_loss=procedure.train(sampeler,dataloader,model,configues,opt,epoch)
    tb_writer.add_scalar("aver_loss",aver_loss,epoch)
    end=time.time()

    print("epoch-"+str(epoch)+":"+str(aver_loss))
    print("----------------------------------------------")



