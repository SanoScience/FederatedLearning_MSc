#!/bin/bash 


echo "server node: $1"
i=0

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	# $1 -> server's node name; $2 -> client's id; $3 -> clients number
	sbatch v100_client.sh $1 $i $2
	i=$((i + 1))
done

 
