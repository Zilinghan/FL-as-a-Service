import copy
import time
import threading
import torch.nn as nn
from ..misc import *
from ..algorithm import *
from funcx import FuncXClient
from omegaconf import DictConfig
from .funcx_server import APPFLFuncXServer
from appfl.funcx.cloud_storage import LargeObjectWrapper
from .funcx_client import client_training, client_testing, client_validate_data

class APPFLFuncXAsyncServer(APPFLFuncXServer):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        super(APPFLFuncXAsyncServer, self).__init__(cfg, fxc)
        cfg["logginginfo"]["comm_size"] = 1
        # Save the version of global model initialized at each client 
        self.client_init_step ={i : 0 for i in range(self.cfg.num_clients)}
        # Create a lock object for the critical section (global model update)
        self.lock = threading.Lock()

    def _initialize_server_model(self):
        """ Initialize the federated learning (APPFL) server. """
        self.server  = eval(self.cfg.fed.servername)(
            self.weights, copy.deepcopy(self.model), self.loss_fn, self.cfg.num_clients, "cpu", **self.cfg.fed.args    
        )
        # Server model should stay on CPU for serialization
        self.server.model.to("cpu")

    def run(self, model: nn.Module, loss_fn: nn.Module):
        # TODO: combine into one run function
        # Set model, and loss function
        self._initialize_training(model, loss_fn)
        # Validate data at clients
        self._validate_clients_data()
        # Calculate weight
        self._set_client_weights(mode=self.cfg.fed.args.client_weights)
        # Initialze model at server
        self._initialize_server_model()
        # Do training
        self._do_training()
        # Do client testing
        self._do_client_testing()
        # Wrap-up
        self._finalize_experiment()
        # Shutdown all clients
        self.trn_endps.shutdown_all_clients()

    def _do_training(self):
        ## Get current global state
        stop_aggregate     = False
        start_time         = time.time()
        def global_update(res, client_idx, client_log):
            client_results = {client_idx: res}
            local_states = [client_results]
            # TODO: fix local update time
            self.cfg["logginginfo"]["LocalUpdate_time"]  = 0
            self.cfg["logginginfo"]["PerIter_time"]      = 0
            self.cfg["logginginfo"]["Elapsed_time"]      = 0
            client_log_dict = OrderedDict()
            client_log_dict[client_idx] = client_log
            self._do_client_validation(self.server.global_step, client_log_dict)
            # self.cfg["logginginfo"]["Validation_time"]   = 0

            # Acquire the global-update lock
            self.logger.info("Acquiring the lock for global update...")
            lock_acquire_time = time.time()
            self.lock.acquire()
            self.logger.info(f"Acquired the lock for global update after {time.time()-lock_acquire_time} sec.")
            # Perform global update
            global_update_start = time.time()
            init_step = self.client_init_step[client_idx]
            self.server.update(local_states, init_step = init_step, client_idx=client_idx)
            global_step = self.server.global_step
            # Logging
            self.cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start
            self.logger.info("Async FL global model updated. GLB step = %02d | Staleness = %02d" % (global_step, global_step - init_step - 1))
            # Save new init step of client
            self.client_init_step[client_idx] = self.server.global_step
            self._lr_step(self.server.global_step)
            # Register new asynchronous task for clients
            self.trn_endps.register_async_func(
                client_training, 
                self.weights, self.server.model.state_dict(), self.loss_fn,
                do_validation = self.cfg.client_do_validation
            )
            # Release the global-update lock
            self.logger.info("Releasing the lock after critical section...")
            self.lock.release()
            # Training eval log
            self._do_server_validation(global_step-1)
            self.server.logging_iteration(self.cfg, self.logger, global_step - 1)
            # Saving checkpoint
            self._save_checkpoint(global_step -1)

        def stopping_criteria():
            return self.server.global_step >= self.cfg.num_epochs
        
        # Register callback function: global_update
        self.trn_endps.register_async_call_back_func(global_update)
        # Register async function: client training
        self.trn_endps.register_async_func(
            client_training, 
            self.weights, self.server.model.state_dict(), self.loss_fn,
            do_validation = self.cfg.client_do_validation)
        # Register the stopping criteria
        self.trn_endps.register_stopping_criteria(stopping_criteria)
        # Start asynchronous FL
        start_time = time.time()
        self.trn_endps.run_async_task_on_available_clients()
        # Run async event loop
        while (not stop_aggregate):
            time.sleep(5)            
            # Define some stopping criteria
            if stopping_criteria():
                self.logger.info("Training is finished!")
                stop_aggregate = True
        self.cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
        return