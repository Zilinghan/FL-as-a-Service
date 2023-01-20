#!/bin/bash
#SBATCH --mem=32g                   # required number of memory
#SBATCH --nodes=1                   # number of required nodes
#SBATCH --ntasks-per-node=1         # number of tasks per node
#SBATCH --cpus-per-task=4           # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu             # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbke-delta-cpu    # <- one of: bbke-delta-cpu or bbke-delta-gpu
#SBATCH --job-name=FaaSMPI          # job name
#SBATCH --time=00:10:00             # dd-hh:mm:ss for the job
#SBATCH --constraint="projects"     # file system dependency: we put the dataset into /projects
#SBATCH --mail-user=zl52@illinois.edu
#SBATCH --mail-type=ALL
### GPU options ###
##SBATCH --gpus-per-node=2
##SBATCH --gpu-bind=none     # <- or closest

 
module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_cpu
module list  # job documentation and metadata
conda init bash # initialize the conda
source ~/.bashrc # reload bashrc to make above command take effect
conda activate appfl
srun mpiexec -np 3 --oversubscribe python mnist_mpi.py --project_dir /projects/bbke/zl52/appfl --num_clients 2
