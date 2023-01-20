## How to Run the MPI Example on Delta Cluster

### System Access
```
$ ssh username@login.delta.ncsa.illinois.edu
```
### Conda Setup
**Skip if finished**
```
$ module load anaconda3_cpu
$ conda init bash
$ bash
```
**Note**: You may see error messages from `conda init bash`, just control-c through them and continue. As long as conda added code to then end of the `~/.bashrc` file, then things will work properly.
### Installation
**Skip is finished**

First create a virtual environment.
```
$ conda create -n APPFL python=3.10.6
$ conda activate APPFL
```
Then install the packages.
```
$ pip install "appfl[analytics]"
$ pip install protobuf==3.20
$ pip install torchvision
$ pip install mpi4py
```
### Submit Batch Job
You just need to do the Conda Setup and Installation once, and then, you can submit your slurm batch by simplying going to the folder containing the script and run:
```
$ sbatch mpi.sh
```
### Useful Reference Link
1) [Delta User Guide](https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide)
2) [Set Anaconda Environment](https://wiki.ncsa.illinois.edu/display/DSC/customizing+Delta+Open+OnDemand)
3) [Slurm Introduction](https://slurm.schedmd.com/quickstart.html)
4) [Example: Run Python Code in Slurm-based Cluster](http://homeowmorphism.com/2017/04/18/Python-Slurm-Cluster-Five-Minutes)