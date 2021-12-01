#!/bin/bash

echo "server node: $1"
i=0

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	sbatch client_classification.sh --node_name=$1 --client_id=$i --clients_number=$2
	i=$((i + 1))
done


