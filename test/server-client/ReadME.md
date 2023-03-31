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

Clone this repository, and go to `test/server-client` folder where contains the benchmarking code.
```
git clone git@github.com:Zilinghan/FL-as-a-Service.git gRPC
cd gRPC/test/server-client
```


### Start the FL Server
Run the following command on one terminal/machine. Specifically, `num_clients` is the total number of clients involved in the federated learning. After running the command below, the console should print the IP address and the port (default 8000) of the FL server, **please note down that IP.**

```
python mnist_grpc_server.py --num_clients=2
```

**Note: you should make sure that the port 8000 is publicly available (allows inbound traffic from 0.0.0.0/0) for your IP. To simplify the process, I recommend to use an EC2 instance which allows all TCP inbound traffic for port 8000. I provide this additional [tutorial](EC2.md) for setting the EC2 server up.**

### Start the FL Clients
Run the following commands on some other terminals/machines. Specifically, `host` is the server IP address, `num_clients` is the total number of clients involved in the federated learning, `client_id` is the 0-indexed id of the client.
```
python mnist_grpc_client.py --client_id=0 --num_clients=2 --host=<SERVER_IP>
```

```
python mnist_grpc_client.py --client_id=1 --num_clients=2 --host=<SERVER_IP>
```

### Start the FL Clients using Batch Job
If you are using HPC or Cluster which uses batch job to allocate the computing resources, you can submit a shell script to run the above commands. Here are two templates [[CPU]](run_client_cpu.sh)/[[GPU]](run_client_gpu.sh) to allocate demanded resources on NCSA Delta cluster. **Remember to change the `client_id` and `num_clients` and `host` accordingly in the shell script.**