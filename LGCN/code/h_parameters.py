import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="LGCN")
    parser.add_argument("--layer",type=int,default=4,
                       help="the layer number of LGCN")
    parser.add_argument("--lr",type=float,default=0.035,
                       help="the learning rate of model")
    parser.add_argument("--epochs",type=int,default=200,
                       help="the epochs of train")
    parser.add_argument("--topk",type=int,default=5,
                       help="the item number of recommendation")
    parser.add_argument("--seed",type=int,default=2023,
                       help="random seed")
    parser.add_argument("--train_batch",type=int,default=32768,
                       help="batc size of train")
    parser.add_argument("--test_batch",type=int,default=200,
                       help="batc size of test")
    parser.add_argument("--embedding_dim",type=int,default=128,
                       help="embedding size of LGCN")
    parser.add_argument("--keep_prob",type=float,default=1,
                       help="the rate of drop_out")
    parser.add_argument("--weight_decay",type=float,default=0.0001,
                       help="weight decay for reg_loss")
    return parser.parse_args()