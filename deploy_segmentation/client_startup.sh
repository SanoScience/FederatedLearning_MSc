sudo /opt/deeplearning/install-driver.sh

# setup metrics
sudo mkdir -p /opt/google
cd /opt/google
sudo git clone https://github.com/GoogleCloudPlatform/compute-gpu-monitoring.git
cd /opt/google/compute-gpu-monitoring/linux
sudo python3 -m venv venv
sudo venv/bin/pip install wheel
sudo venv/bin/pip install -Ur requirements.txt
sudo cp /opt/google/compute-gpu-monitoring/linux/systemd/google_gpu_monitoring_agent_venv.service /lib/systemd/system
sudo systemctl daemon-reload
sudo systemctl --no-reload --now enable /lib/systemd/system/google_gpu_monitoring_agent_venv.service

# prepare FL
cd /home/prz_jab98
gsutil cp gs://fl-msc-segmentation-dataset/chest_dataset.zip .
unzip chest_dataset.zip

git clone https://${var.token}@github.com/SanoScience/FederatedLearning_MSc.git
sudo chmod -R 777 FederatedLearning_MSc
cd FederatedLearning_MSc/segmentation

CURR_DIR=$PWD
PARENT_DIR="$(dirname "$CURR_DIR")"
export PYTHONPATH=$PARENT_DIR
touch test.txt
python3 client_segmentation.py ${google_compute_address.flower-server.address} ${count.index} ${var.node_count} > logs.txt