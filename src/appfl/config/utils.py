import importlib.util
import inspect
from ..models import *
from omegaconf import OmegaConf
from .config import *
import yaml
import os.path as osp
from ..misc.utils import *
def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))


def load_exct_func_cfg(cfg_dict):
    src = OmegaConf.create(
        ExecutableFunc(**cfg_dict))
    assert src.module != "" or  src.script_file  != "",  "Need to specify the executable function by (module, call) or script file"
    assert not (src.module != "" and src.script_file != ""), "Can only specify the executable function by (module, call) or script file but not both"
    assert src.call != "", "Need to specify the function's name by setting 'call: <func name>' in the config file"
    if src.script_file != "":
        with open(src.script_file) as fi:
            src.source = fi.read()
        assert len(src.source) > 0, "Source file is empty."
    return src

def load_funcx_config(cfg: FuncXConfig,
    config_file: str):
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)

    ## Load module configs for get_model and get_dataset method
    if 'get_data' in data['func']:
        cfg.get_data = load_exct_func_cfg(data['func']['get_data'])
    
    if 'loss' in data:
        cfg.loss = data['loss']
    if 'train_data_batch_size' in data:
        cfg.train_data_batch_size = data['train_data_batch_size']
    if 'test_data_batch_size' in data:
        cfg.test_data_batch_size = data['test_data_batch_size']
    cfg.get_model= load_exct_func_cfg(data['func']['get_model'])
    
    ## Load FL algorithm configs
    cfg.fed =Federated()
    cfg.fed.servername = data['algorithm']['servername']
    cfg.fed.clientname = data['algorithm']['clientname']
    cfg.fed.args = OmegaConf.create(data['algorithm']['args'])
    ## Load training configs
    cfg.num_epochs         = data['training']['num_epochs']
    if 'save_model_dirname' in data['training']:
        cfg.save_model_dirname = data['training']['save_model_dirname']
    cfg.save_model_filename= data['training']['save_model_filename']
    ## Load model configs
    cfg.model_kwargs       = data['model']
    ## Load dataset configs
    cfg.dataset  = data['dataset']['name']
    
def load_funcx_device_config(cfg: FuncXConfig, 
    config_file: str):
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    
    ## Load configs for server
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))
    
    ## Load configs for clients
    for client in data["clients"]:
        if 'get_data' in client:
            client['get_data'] = load_exct_func_cfg(client['get_data'])
        if 'data_pipeline' in client:
            client['data_pipeline']= OmegaConf.create(client['data_pipeline'])
        # else:
        #     client['data_pipeline']= OmegaConf.create({})
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    
    cfg.num_clients = len(cfg.clients)
    return cfg    

#====================================================================================


def load_appfl_server_config_funcx(cfg: FuncXConfig, config_file: str):
    # Modified (ZL): load the configuration file for the appfl server
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)

    ## Load configs for server
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))

    ## Load module configs for get_model and get_dataset method
    if 'get_data' in data['func']:
        cfg.get_data = load_exct_func_cfg(data['func']['get_data'])
    
    if 'loss' in data:
        cfg.loss = data['loss']
    if 'train_data_batch_size' in data:
        cfg.train_data_batch_size = data['train_data_batch_size']
    if 'test_data_batch_size' in data:
        cfg.test_data_batch_size = data['test_data_batch_size']
    cfg.get_model= load_exct_func_cfg(data['func']['get_model'])
    
    ## Load FL algorithm configs
    cfg.fed =Federated()
    cfg.fed.servername = data['algorithm']['servername']
    cfg.fed.clientname = data['algorithm']['clientname']
    cfg.fed.args = OmegaConf.create(data['algorithm']['args'])
    ## Load training configs
    cfg.num_epochs         = data['training']['num_epochs']
    if 'save_model_dirname' in data['training']:
        cfg.save_model_dirname = data['training']['save_model_dirname']
    cfg.save_model_filename= data['training']['save_model_filename']
    ## Load model configs
    cfg.model_kwargs       = data['model']
    ## Load dataset configs
    cfg.dataset  = data['dataset']['name']
    
def load_appfl_client_config_funcx(cfg: FuncXConfig, config_file: str):
    # Modified (ZL): Load the configuration file for appfl clients
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    
    ## Load configs for clients
    for client in data["clients"]:
        if 'get_data' in client:
            client['get_data'] = load_exct_func_cfg(client['get_data'])
        if 'data_pipeline' in client:
            client['data_pipeline']= OmegaConf.create(client['data_pipeline'])
        # else:
        #     client['data_pipeline']= OmegaConf.create({})
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    
    cfg.num_clients = len(cfg.clients)
    return cfg

def get_call(script: str):
    '''
    return the name of the first function inside a python script
    '''
    module_spec = importlib.util.spec_from_file_location('module', script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    functions = inspect.getmembers(module, inspect.isfunction)
    function_names = [func[0] for func in functions]
    assert len(function_names) == 1, f"More than one function defined in script {script}"
    return function_names[0]

def load_appfl_client_config_funcx_web(cfg: FuncXConfig, config_files: List[str], dataloaders: List[str]):
    assert len(config_files) == len(dataloaders), "The number of configuration files and dataloader files are different!"
    for config_file, dataloader_file in zip(config_files, dataloaders):
        assert osp.exists(config_file), f"Config file {config_file} not found!"
        assert osp.exists(dataloader_file), f"Dataloader file {dataloader_file} not found!"
        # load the client configuration file
        with open(config_file) as fi:
            data = yaml.load(fi, Loader = yaml.SafeLoader)
        client = data['client']
        # load the client dataloader
        src =  OmegaConf.create(ExecutableFunc())
        src.script_file = dataloader_file
        with open(dataloader_file) as fi:
            src.source = fi.read()
        src.call = get_call(dataloader_file)
        client['get_data'] = src
        # add the client
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    cfg.num_clients = len(cfg.clients)
    return cfg

def load_appfl_server_config_funcx_web(cfg: FuncXConfig, server_config: str, model_config: str, model_file: str):
    assert osp.exists(server_config), f"Config file {server_config} not found!"
    assert osp.exists(model_config), f"Config file {model_config} not found!"
    assert osp.exists(model_file), f"Model loader {model_file} not found!"

    # Load server configuration file
    with open(server_config) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))
    ## Load module configs for get_dataset method
    if 'get_data' in data:
        cfg.get_data = load_exct_func_cfg(data['func']['get_data'])
    if 'loss' in data:
        cfg.loss = data['loss']
    if 'train_data_batch_size' in data:
        cfg.train_data_batch_size = data['train_data_batch_size']
    if 'test_data_batch_size' in data:
        cfg.test_data_batch_size = data['test_data_batch_size']

    ## Load the dataloader
    src = OmegaConf.create(ExecutableFunc())
    src.script_file = model_file
    with open(model_file) as fi:
        src.source = fi.read()
    src.call = get_call(model_file)
    cfg.get_model = src
    
    ## Load FL algorithm configs
    cfg.fed =Federated()
    cfg.fed.servername = data['algorithm']['servername']
    cfg.fed.clientname = data['algorithm']['clientname']
    cfg.fed.args = OmegaConf.create(data['algorithm']['args'])
    ## Load training configs
    cfg.num_epochs         = data['training']['num_epochs']
    if 'save_model_dirname' in data['training']:
        cfg.save_model_dirname = data['training']['save_model_dirname']
    cfg.save_model_filename= data['training']['save_model_filename']
    ## Load dataset configs
    cfg.dataset  = data['dataset']['name']
  
    ## Load model configs
    with open(model_config) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    cfg.model_kwargs = data['model']


def load_appfl_server_config_funcx_web_v2(cfg: FuncXConfig, server_config: str):
    assert osp.exists(server_config), f"Config file {server_config} not found!"
    # assert osp.exists(model_config), f"Config file {model_config} not found!"
    # assert osp.exists(model_file), f"Model loader {model_file} not found!"

    # Load server configuration file
    with open(server_config) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))
    ## Load module configs for get_dataset method
    if 'get_data' in data:
        cfg.get_data = load_exct_func_cfg(data['func']['get_data'])
    if 'loss' in data:
        cfg.loss = data['loss']
    if 'train_data_batch_size' in data:
        cfg.train_data_batch_size = data['train_data_batch_size']
    if 'test_data_batch_size' in data:
        cfg.test_data_batch_size = data['test_data_batch_size']

    ## Load the model
    src = OmegaConf.create(ExecutableFunc())
    model_dict = {'CNN': CNN.__file__}
    # src.module = model_dict[data['model_type']]
    # print(CNN.__file__)
    # print(os.path.abspath(CNN.__file__))
    # src.call = 'get_model'
    src.script_file = model_dict[data['model_type']]
    with open(src.script_file) as fi:
        src.source = fi.read()
    src.call = get_call(src.script_file)
    cfg.get_model = src
    
    ## Load FL algorithm configs
    cfg.fed =Federated()
    cfg.fed.servername = data['algorithm']['servername']
    cfg.fed.clientname = data['algorithm']['clientname']
    cfg.fed.args = OmegaConf.create(data['algorithm']['args'])
    ## Load training configs
    cfg.num_epochs         = data['training']['num_epochs']
    # TODO: Currently, the save model is disabled
    if 'save_model_dirname' in data['training']:
        cfg.save_model_dirname = data['training']['save_model_dirname']
    cfg.save_model_filename= data['training']['save_model_filename']
    ## Load dataset configs
    cfg.dataset  = data['dataset']['name']
  
    ## Load model configs
    cfg.model_kwargs = data['model']

