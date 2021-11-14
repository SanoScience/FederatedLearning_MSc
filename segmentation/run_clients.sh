#!/bin/bash 


echo "server node: $1"
i=0

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	sbatch client.sh $1 $i $2
	i=$((i + 1))
done

 
