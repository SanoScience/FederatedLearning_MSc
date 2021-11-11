#!/bin/bash 


echo "server node: $1"

for i in {1..$2}
do 
    echo "starting $i/$2 client"
    sbatch client.sh $1 $i $2
done
