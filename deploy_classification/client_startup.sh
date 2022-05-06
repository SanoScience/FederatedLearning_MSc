# Enable persistence mode on all GPUs (to set utilization to 0%)
sudo nvidia-smi -pm 1

git clone -b add-classification https://${token}@github.com/SanoScience/FederatedLearning_MSc.git
sudo chmod -R 777 FederatedLearning_MSc
cd FederatedLearning_MSc/classification

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ffcv

sleep 30
echo 'Running FL'
python3 client_classification.py --sa ${address} --c_id ${index} --c ${node_count} --d ${client_dataset} > output.txt
