# How to Run funcX FL on Delta Cluster

## Installation
0. [For Delta] Load the anaconda module 

    ```
    module load anaconda3_gpu
    ```
1. Create a virtual environment using `conda`

    ```
    conda create -n funcx python=3.8
    conda activate funcx
    ```
2. Clone this repository [**Note**: If you find some packages unable to install, simple `Ctrl+C` to skip them. I adopt the original requirements file, which is not a perfect one.]

    ```
    git clone https://github.com/Zilinghan/FL-as-a-Service.git FaaS
    cd FaaS
    git checkout funcx
    pip install -r requirements.txt
    pip install -e .
    pip install funcx-endpoint
    ```


## FuncX Endpoint Config
3. Setup funcX endpoint. Please add your own `<ENDPOINT_NAME>` such as `delta-gpu`. 

    You might be required to login with [Globus](https://app.globus.org), following the prompt instructions and finish the authentication steps.

    ```
    funcx-endpoint configure <ENDPOINT_NAME>
    ```

4. Confiugre the endpoint by editting the file `~/.funcx/<ENDPOINT_NAME>/config.py`. A sample file can be found [here](funcx/endpoint_configs/config.py). Please pay attention to the following points:

    (1) Put whatever cmds you want to run before starting a worker into `'worker_init'` part.

    (2) Put whatever cmds you want to run with `#SBATCH` into the `'scheduler_options'` part, e.g., change the `--mail-user` to your email address.
    
    (3) Replace the `<ENDPOINT_NAME>` to your created name.

5. Start the funcX endpoint. The follwoing command will allocate resources you required from the `config.py` file above. [**Note**: Whenever you modify the `config.py`, you need to first run `funcx-endpoint stop <ENDPOINT_NAME>` and then re-start it to have the changes make effect.]

    ```
    funcx-endpoint start <ENDPOINT_NAME>
    ```

## Submit Batch Job using Slurm
6. Obtain your created endpoint ID by running the following command. Then copy and paste the ID to the corresponding field of `configs/clients/mnist_broad.yaml`.

    ```
    funcx-endpoint list
    ```

7. Check the `run_funcx_sync.sh` file, and make necessary modifications such as changing the `--mail-user`. Then you can submit the job

    ```
    sbatch run_funcx_sync.sh
    ```

