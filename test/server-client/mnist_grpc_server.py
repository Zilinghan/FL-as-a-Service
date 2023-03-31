import sys
import time
import socket
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

parser.add_argument('--dataset', type=str, default="MNIST")             # Only used for logging, @Hieu: you can change it to COVID/ECG  

## clients
parser.add_argument('--num_clients', type=int, default=2)               # number of training clients

## server
# @Hieu: Please make sure that the server uses the same federation algorithm as APPFLx
parser.add_argument('--server', type=str, default="ServerFedAvg")       # name of the server algorithm     
parser.add_argument('--num_epochs', type=int, default=2)                # number of server global training epochs
parser.add_argument('--validation', type=bool, default=True)            # whether to use server validation: @Hieu, if COVID/ECG do not have server validation dataset, set this to False
 
args = parser.parse_args()    

if torch.cuda.is_available():
    args.device="cuda"
 
# @Hieu: if you are not doing validation/test on the server side, safely ignore this
def get_test_data():
    # test data for a server
    test_data_raw = eval("torchvision.datasets.MNIST")(
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

# @Hieu: Please change this get_model function to load your model
def get_model():
    ## User-defined model
    model = CNN(1, 10, 28)
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

    # get the local IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    cfg.server.host = ip_address
    cfg.server.port = 8000
    print(f"Server is running on {cfg.server.host}:{cfg.server.port}...")

    ## outputs        
    cfg.output_dirname = "./outputs_%s_%s"%(args.dataset, args.server)     
    cfg.output_filename = "result"    

    start_time = time.time()

    """ User-defined model """    
    model = get_model()
    loss_fn = torch.nn.CrossEntropyLoss()  #@Hieu: check the loss function as well

    ## loading and saving models 
    cfg.load_model = False
    cfg.save_model = False

    """ User-defined data """        
    if cfg.validation:
        test_dataset = get_test_data()   
    else:
        test_dataset = Dataset()   
 
    print("-------Loading_Time=", time.time() - start_time)    
 
    grpc_server.run_server(cfg, model, loss_fn, args.num_clients, test_dataset)

if __name__ == "__main__":
    main()
