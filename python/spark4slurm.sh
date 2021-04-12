#! /bin/bash

if [ $SLURM_NODEID = 0 ]; then
#    echo " I am Master"
    MASTER=$(hostname)
    export SPARK_MASTER_IP=$(hostname)
    echo "Master:" $SPARK_MASTER_IP "SLURM_NODEID" $SLURM_NODEID

    "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
        --ip $SPARK_MASTER_IP \
        --port 7077 \
        --webui-port 8080
else
#    echo "I am slave"
#    echo $(hostname)
    MASTER_NODE=spark://$(scontrol show hostname $SLURM_NODELIST | head -n 1):7077
    echo "slave" $(hostname) "-> master:" $MASTER_NODE "SLURM_NODEID" $SLURM_NODEID
    "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker $MASTER_NODE
fi


