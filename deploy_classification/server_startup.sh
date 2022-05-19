# Enable persistence mode on all GPUs (to set utilization to 0%)
sudo nvidia-smi -pm 1

cd /home/filip_slazyk/

git clone -b add-classification https://${token}@github.com/SanoScience/FederatedLearning_MSc.git
sudo chmod -R 777 FederatedLearning_MSc
cd FederatedLearning_MSc/classification

source /home/filip_slazyk/anaconda3/etc/profile.d/conda.sh
conda activate ffcv

echo 'Running FL'
python3 server_classification.py --c ${node_count} --r ${rounds} --m ${model} --d ${training_datasets} --le ${local_epochs} --lr ${learning_rate} --bs ${batch_size} --mf ${min_fit_clients} --ff ${fraction_fit} --data-selection ${data_selection} --test_datasets ${test_datasets} --results_bucket ${results_bucket} --study_prefix ${study_prefix}
