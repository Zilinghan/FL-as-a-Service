import torch
import argparse
from models.cnn import *
from appfl.config import *
from funcx import FuncXClient
from appfl.misc.data import *
from appfl.misc.utils import *
from appfl.misc.logging import *
import appfl.run_funcx_server as funcx_server

parser = argparse.ArgumentParser()  
parser.add_argument("--client_config", type=str, default="configs/clients/client.yaml")
parser.add_argument("--config", type=str, default= "configs/fed_avg/funcx_fedavg_mnist.yaml") 
parser.add_argument('--clients-test', action='store_true', default=False)
parser.add_argument('--reproduce', action='store_true', default=True) 
parser.add_argument('--load-model', action='store_true', default=False) 
parser.add_argument("--load-model-dirname", type=str, default= "")
parser.add_argument("--load-model-filename", type=str, default= "")
parser.add_argument('--use-tensorboard', action='store_true', default=True)
args = parser.parse_args()

def main():
    """ Configuration """     
    cfg = OmegaConf.structured(FuncXConfig)
    cfg.reproduce = True
    cfg.save_model_state_dict = True
    cfg.save_model = True
    cfg.checkpoints_interval = 2
    cfg.load_model = args.load_model
    cfg.load_model_dirname  = args.load_model_dirname
    cfg.load_model_filename = args.load_model_filename
    if cfg.reproduce == True:
        set_seed(1)

    ## execution mode
    mode = 'clients_testing' if args.clients_test else 'train'

    ## loading funcX configs from file
    load_funcx_device_config(cfg, args.client_config)
    load_funcx_config(cfg, args.config)

    ## using funcx ClientOptimizer object
    cfg.fed.clientname = "FuncxClientOptim"
    
    ## tensorboard
    cfg.use_tensorboard= args.use_tensorboard
    
    ## config logger
    mLogging.config_logger(cfg)

    ## validation
    cfg.validation = True   
    
    """ Server-defined model """
    ModelClass     = get_executable_func(cfg.get_model)()
    model          = ModelClass(*cfg.model_args, **cfg.model_kwargs) 
    loss_fn        = get_loss_func(cfg.loss)

    if cfg.load_model == True:
        path = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
        print("Loading model from ", path)
        model.load_state_dict(torch.load(path)) 
        model.eval()

    """ User-defined data """
    logger = mLogging.get_logger()

    """ Prepare test dataset"""
    server_test_dataset = None
    server_val_dataset  = None

    """ APPFL with funcX """
    ## create funcX client object
    fxc = FuncXClient(force_login=True)
    ## run funcX server
    funcx_server.run_server(cfg, model, loss_fn, fxc, server_test_dataset, server_val_dataset, mode=mode)

if __name__ == "__main__":
    main()