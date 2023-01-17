## How to run the server-client test

### Installation
First create a virtual environment.
```
conda create -n APPFL python=3.10.6
conda activate APPFL
```
Then install the packages.
```
pip install "appfl[analytics]"
pip install protobuf==3.20
```


### Server
Run the following command on one terminal/machine. Specifically, `host` is the IP address of the server, `num_clients` is the total number of clients involved in the federated learning.
```
python mnist_grpc_server.py --host=192.168.10.50 --num_clients=2
```

### Client
Run the following commands on some other terminals/machines (if using different machines, currently, it is required that they should be connected to the same WiFi). Specifically, `host` is the server IP address, `num_clients` is the total number of clients involved in the federated learning, `client_id` is the 0-index id of the client.
```
python mnist_grpc_client.py --host=192.168.10.50 --client_id=0 --num_clients=2
```

```
python mnist_grpc_client.py --host=192.168.10.50 --client_id=1 --num_clients=2
```