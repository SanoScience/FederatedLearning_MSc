#!/bin/bash

echo "server node: $1"
echo "server job id: $4"
i=0
touch "$4_client_ids.txt"

while [ "$i" -lt "$2" ]; do
	echo "starting $i/$2 client"
	sbatch client_classification.sh $1 $i $2 $3 | cut -d " " -f 4 >> $4
	i=$((i + 1))
done