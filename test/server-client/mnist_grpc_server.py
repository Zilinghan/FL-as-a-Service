import sys
import time
import logging
import argparse
sys.path.append('../..')

import torch
import torchvision
from torchvision.transforms import ToTensor

from models.cnn import *
from src.appfl.config import *
from src.appfl.misc.data import *
from src.appfl.misc.utils import *
import src.appfl.run_grpc_server as grpc_server


"""
python mnist_grpc_server.py --host=192.168.10.50 --num_clients=2
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
parser.add_argument('--num_clients', type=int, default=2)    

## server
parser.add_argument('--server', type=str, default="ServerFedAvg")    
parser.add_argument('--num_epochs', type=int, default=2) 
parser.add_argument('--validation', type=bool, default=True)     
parser.add_argument('--host', type=str, default="localhost") 
 
args = parser.parse_args()    

if torch.cuda.is_available():
    args.device="cuda"
 
 
def get_test_data():
    # test data for a server
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

    return test_dataset


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
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    cfg.server.host = args.host

    ## outputs        
    cfg.output_dirname = "./outputs_%s_%s"%(args.dataset, args.server)     
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
    test_dataset = get_test_data()
    
    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check_testset(test_dataset, args.num_channel, args.num_pixel)        
 
    print(
        "-------Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "MNIST_CNN"    
 
    grpc_server.run_server(cfg, model, loss_fn, args.num_clients, test_dataset)

if __name__ == "__main__":
    main()
