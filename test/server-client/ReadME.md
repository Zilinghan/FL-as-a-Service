## Benchmarking APPFL with gRPC

### Installation
First create a virtual environment.
```
conda create -n grpc python=3.8
conda activate grpc
```
Then install the packages.
```
pip install "appfl[analytics]"
pip install protobuf==3.20
pip install torchvision
```


### Start the Server
Run the following command on one terminal/machine. Specifically, `num_clients` is the total number of clients involved in the federated learning. After running the command below, the console should print the IP address of the server, **please note down that IP.**
```
python mnist_grpc_server.py --num_clients=2
```

### Start the Clients
Run the following commands on some other terminals/machines. Specifically, `host` is the **server IP address**, `num_clients` is the total number of clients involved in the federated learning, `client_id` is the 0-indexed id of the client.
```
python mnist_grpc_client.py --client_id=0 --num_clients=2 --host=<SERVER_IP>
```

```
python mnist_grpc_client.py --client_id=1 --num_clients=2 --host=<SERVER_IP>
```