import torch
import argparse
import torchvision
import os.path as osp
from models.cnn import *
from appfl.config import *
from funcx import FuncXClient
from appfl.misc.data import *
from appfl.misc.utils import *
from appfl.misc.logging import *
from torchvision.transforms import ToTensor
import appfl.run_funcx_async_server as funcx_async_server

parser = argparse.ArgumentParser()  
parser.add_argument("--device_config", type=str,  default='./configs/clients/client.yaml')
parser.add_argument("--config", type=str, default='./configs/fed_async/funcx_fedasync_mnist.yaml') 
parser.add_argument('--reproduce', action='store_true', default=True) 
parser.add_argument('--use_tensorboard', action='store_true', default=True)
args = parser.parse_args()

def main():
    """ Configuration """     
    cfg = OmegaConf.structured(FuncXConfig)
    if cfg.reproduce == True:
        set_seed(1)

    ## loading funcX configs from file
    load_funcx_device_config(cfg, args.device_config)
    load_funcx_config(cfg, args.config)

    ## tensorboard
    cfg.use_tensorboard= args.use_tensorboard
    
    ## config logger
    mLogging.config_logger(cfg)

    ## validation
    cfg.validation = True   
    
    """ User-defined model """
    ModelClass     = get_executable_func(cfg.get_model)()
    model          = ModelClass(*cfg.model_args, **cfg.model_kwargs) 
    loss_fn        = torch.nn.CrossEntropyLoss()  

    """ User-defined data """
    # Prepare dataset for testing at server 
    test_data_input = []
    test_data_label = []

    test_data_raw = eval("torchvision.datasets." + cfg.dataset)(
            osp.join(cfg.server.data_dir, "RawData"), download=True, train=False, transform=ToTensor()
        )

    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    """ APPFL with funcX """
    ## create funcX client object
    fxc = FuncXClient(batch_enabled=True)
    ## run funcX server
    funcx_async_server.run_server(cfg, model, loss_fn, fxc , test_dataset)

if __name__ == "__main__":
    main()