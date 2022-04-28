#!/bin/bash

echo "server node: $1"
echo "clients count: $2"
echo "server job id: $3"
echo "dataset: $4"

i=0
touch "$3_client_ids.txt"

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	sbatch client_classification.sh $1 $i $2 $4 | cut -d " " -f 4 >> "$3_client_ids.txt"
	i=$((i + 1))
done