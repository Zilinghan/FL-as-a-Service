import sys
import time
import logging
import argparse
sys.path.append('../..')

import torch
import torchvision
import numpy as np
from torchvision.transforms import ToTensor

from models.cnn import *
from src.appfl.config import *
from src.appfl.misc.data import *
from src.appfl.misc.utils import *
import src.appfl.run_grpc_client as grpc_client

"""
python mnist_grpc_client.py --host=192.168.10.50 --client_id=0 --num_clients=2
"""

""" read arguments """ 

parser = argparse.ArgumentParser() 

parser.add_argument('--device', type=str, default="cpu")    

## dataset
parser.add_argument('--dataset', type=str, default="MNIST")   
parser.add_argument('--num_channel', type=int, default=1)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=28)   

## clients
parser.add_argument('--num_clients', type=int, default=1)    
parser.add_argument('--client_optimizer', type=str, default="Adam")    
parser.add_argument('--client_lr', type=float, default=1e-3)    
parser.add_argument('--num_local_epochs', type=int, default=3)    
parser.add_argument('--validation', type=bool, default=True)
parser.add_argument('--client_id', type=int, required=True)

## server
parser.add_argument('--host', type=str, default='localhost')

args = parser.parse_args()    

if torch.cuda.is_available():
    args.device="cuda"
 
def get_data():
    test_data_raw = eval("torchvision.datasets." + args.dataset)(
        f"./datasets/RawData", download=True, train=False, transform=ToTensor()
    )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + args.dataset)(
        f"./datasets/RawData", download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset


def get_model():
    ## User-defined model
    model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    return model


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    """ Configuration """     
    cfg = OmegaConf.structured(Config)
    
    cfg.device = args.device 
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)
    cfg.validation = args.validation

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.server.host = args.host
    
    ## outputs        
    cfg.output_dirname = "./outputs_%s_%s"%(args.dataset, args.client_optimizer)     
    cfg.output_filename = "result"    

    start_time = time.time()

    """ User-defined model """    
    model = get_model()
    loss_fn = torch.nn.CrossEntropyLoss()  

    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)      

    """ User-defined data """        
    train_datasets, test_dataset = get_data()
    
    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)        

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "MNIST_CNN"    
     
    grpc_client.run_client(cfg, args.client_id, model, loss_fn, train_datasets[args.client_id], args.client_id, test_dataset)
            
    print("------DONE------", args.client_id)



if __name__ == "__main__":
    main()
