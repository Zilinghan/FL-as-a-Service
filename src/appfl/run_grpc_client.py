from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import copy
import numpy as np
import logging
import time

from .misc import *
from .algorithm import *

from .protos.federated_learning_pb2 import Job
from .protos.client import FLClient


def update_model_state(comm, model, round_number):
    new_state = {}
    for name in model.state_dict():
        nparray = comm.get_tensor_record(name, round_number)
        new_state[name] = torch.tensor(nparray)
    model.load_state_dict(new_state)


def run_client(
    cfg: DictConfig,
    cid: int,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    gpu_id: int = 0,
    test_data: Dataset = Dataset(),
) -> None:
    """Launch gRPC client to connect to the server specified in the configuration.

    Args:
        cfg (DictConfig): the configuration for this run
        cid (int): cliend_id
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function
        train_data (Dataset): training data
        gpu_id (int): GPU ID
    """
    
    
    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    output_filename = cfg.output_filename + "_client_%s" % (cid)
    output_filename, outfile = client_log_new(cfg.output_dirname, output_filename)
    fmt = logging.Formatter('[%(asctime)s %(levelname)-4s]: %(message)s')  
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(output_filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    c_handler.setFormatter(fmt)
    f_handler.setFormatter(fmt)
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    if cfg.server.use_tls == True:
        uri = cfg.server.host
    else:
        uri = cfg.server.host + ":" + str(cfg.server.port)

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        cfg.device = f"cuda:{gpu_id}"
    else:
        cfg.device = cfg.device

    batch_size = cfg.train_data_batch_size
    if cfg.batch_training == False:
        cfg.train_data_batch_size = len(train_data)

    logger.info(
        f"[Client ID: {cid: 03}] connecting to (uri,tls)=({uri},{cfg.server.use_tls})."
    )
    comm = FLClient(
        cid,
        uri,
        cfg.server.use_tls,
        max_message_size=cfg.max_message_size,
        api_key=cfg.server.api_key,
    )

    # Retrieve its weight from a server.
    weight = -1.0
    i = 1
    logger.info(f"[Client ID: {cid: 03}] requesting weight to the server.")
    try:
        while True:
            weight = comm.get_weight(len(train_data))
            logger.info(
                f"[Client ID: {cid: 03}] trial {i}, requesting weight ({weight})."
            )
            if weight >= 0.0:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info(f"[Client ID: {cid: 03}] terminating the client.")
        return

    if weight < 0.0:
        logger.error(f"[Client ID: {cid: 03}] weight ({weight}) retrieval failed.")
        return


    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
        test_dataloader = None

    fed_client = eval(cfg.fed.clientname)(
        cid,
        weight,
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data,
            num_workers=cfg.num_workers,
            batch_size=batch_size,
            shuffle=cfg.train_data_shuffle,
            pin_memory=True,
        ),
        cfg,
        outfile,
        test_dataloader,
        **cfg.fed.args,
    )

    # Start federated learning.
    cur_round_number, job_todo = comm.get_job(Job.INIT)
    prev_round_number = 0
    learning_time = 0.0
    send_time = 0.0
    cumul_learning_time = 0.0

    while job_todo != Job.QUIT:
        if job_todo == Job.TRAIN:
            if prev_round_number != cur_round_number: 
                logger.info(
                    f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Start training"
                )
                update_model_state(comm, fed_client.model, cur_round_number)
                logger.info(
                    f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Received model update from server"
                )
                prev_round_number = cur_round_number

                time_start = time.time()
                local_state = fed_client.update()
                time_end = time.time()

                learning_time = time_end - time_start
                cumul_learning_time += learning_time

                if (
                    cur_round_number % cfg.checkpoints_interval == 0
                    or cur_round_number == cfg.num_epochs
                ):
                    """Saving model"""
                    if cfg.save_model == True:
                        save_model_iteration(cur_round_number, fed_client.model, cfg)

                time_start = time.time()
                comm.send_learning_results(
                    local_state["penalty"],
                    local_state["primal"],
                    local_state["dual"],
                    cur_round_number,
                )
                time_end = time.time()
                send_time = time_end - time_start
                logger.info(
                    f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Trained (Time %.4f, Epoch {cfg.fed.args.num_local_epochs: 03}) and sent results back to the server (Elapsed %.4f)",
                    learning_time,
                    send_time,
                )
            else:
                logger.info(
                    f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Waiting for next job"
                )
                time.sleep(5)
        cur_round_number, job_todo = comm.get_job(job_todo)
        if job_todo == Job.QUIT:
            logger.info(
                f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Quitting... Learning %.4f Sending %.4f Receiving %.4f Job %.4f Total %.4f",
                cumul_learning_time,
                comm.time_send_results,
                comm.time_get_tensor,
                comm.time_get_job,
                comm.get_comm_time(),
            )
            # Update with the most recent weights before exit.
            update_model_state(comm, fed_client.model, cur_round_number)

            outfile.close()


if __name__ == "__main__":
    run_client()
