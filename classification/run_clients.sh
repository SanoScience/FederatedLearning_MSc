#!/bin/bash

echo "server node: $1"
i=0

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	sbatch client_classification.sh $1 $i $2 $3
	i=$((i + 1))
done


