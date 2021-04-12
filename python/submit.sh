#!/bin/bash

#SBATCH -N 16 -n 16
#SBATCH --ntasks-per-node=1

srun sh spark4slurm.sh & sleep 10

MASTER=spark://$(scontrol show hostname $SLURM_NODELIST | head -n 1):7077
echo $MASTER_NODE
$SPARK_HOME/bin/spark-submit /home/LAB/zhuxk/project/sparker/python/test.py
#--master $MASTER_NODE 
