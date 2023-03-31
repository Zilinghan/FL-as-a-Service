# Deploy a gRPC server on EC2

1. Create an EC2 instance [here](https://console.aws.amazon.com/ec2/) by referring to this [guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html). **NOTE:** 

    (1) Please select **Amazon Linux 2 AMI (HVM) - Kernel 5.10, SSD Volumn Type** as your OS images. 

    (2) Now I select **t2.micro** as the instance type, which is very cheap for the deployment stage (free-tier eligible). 
    
    (3) For the security group, we should allow TCP inbound traffic for port 8000 from all sources (0.0.0.0/0), and ssh from your own IP address for development. 
    
    (4) Allocate **more than 20GB** of General Purpose SSD (gp2).

2. Connect to your EC2 instance using ssh. `your-pem.pem` is a key you create in this [guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html).
    ```
    $ ssh -i your-pem.pem ec2-user@your-ec2-instance-ipv4dns
    ```

3. Install `git` in your EC2 instance.
    ```
    $ sudo yum update -y
    $ sudo yum install git -y
    ```

4. Generate `ssh-key` for your EC2 instance, then copy the key to Github to create an ssh-key for accessing github inside the EC2 instance.
    ```
    $ ssh-keygen -t rsa -b 4096 -C "your-email@illinois.edu"
    $ cat ~/.ssh/id_rsa.pub
    ```

5. Allocate more memory from SSD using swapfile. Run the following command to create a swap file with a size of 10 GB (you can adjust the size as needed).
    ```
    $ sudo fallocate -l 10G /swapfile
    ```
    Set the correct permissions for the swap file by running the following command:
    ```
    $ sudo chmod 600 /swapfile
    ```
    Set up the swap space on the file by running the following command:
    ```
    $ sudo mkswap /swapfile
    ```
    Enable the swap file by running the following command:
    ```
    $ sudo swapon /swapfile
    ```
    To make the swap file permanent across reboots, add an entry for the swap file to the `/etc/fstab` file. Open the `/etc/fstab` file in a text editor and add the following line at the end of the file:
    ```
    /swapfile swap swap defaults 0 0
    ```

6. Install `conda`.
    ```
    $ mkdir conda
    $ cd conda
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ source ~/.bashrc
    ```

7. Create a virtual environment.
    ```
    conda create -n grpc python=3.8
    conda activate grpc
    ```

8. Install the packages.
    ```
    pip install "appfl[analytics]"
    pip install protobuf==3.20
    pip install torchvision
    ```

9. Clone the repository and start the server. **Note: For EC2, you should not use the printed IP address, which is the private IP, instead, go to EC2 console and use the public IP**
    ```
    cd ~
    git clone git@github.com:Zilinghan/FL-as-a-Service.git gRPC
    cd gRPC/test/server-client
    python mnist_grpc_server.py --num_clients=2
    ```