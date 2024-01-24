from h_parameters import parse_args
from os.path import join
import torch
args=parse_args()
configues={}
configues["layer"]=args.layer
configues["lr"]=args.lr
configues["epochs"]=args.epochs
configues["topk"]=args.topk
configues["seed"]=args.seed
configues["train_batch"]=args.train_batch
configues["test_batch"]=args.test_batch
configues["embedding_dim"]=args.embedding_dim
configues["keep_prob"]=args.keep_prob
configues["weight_decay"]=args.weight_decay


ROOT_PATH="../"
DATA_PATH=join(ROOT_PATH,"data")
CODE_PATH=join(ROOT_PATH,"code")
EXPERIMENT_PATH=join(ROOT_PATH,"experiment")
GPU=torch.cuda.is_available()
device=torch.device('cuda' if GPU else 'cpu')
# device=torch.device('cpu')
configues["ROOT_PATH"]=ROOT_PATH
configues["DATA_PATH"]=DATA_PATH
configues["CODE_PATH"]=CODE_PATH
configues["EXPERIMENT_PATH"]=EXPERIMENT_PATH
configues["device"]=device
