#curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
#sudo python3 install_gpu_driver.py
sudo /opt/deeplearning/install-driver.sh


# setup metrics
#sudo apt-get install python3-venv -y
#sudo mkdir -p /opt/google
#cd /opt/google
#sudo git clone https://github.com/GoogleCloudPlatform/compute-gpu-monitoring.git
#cd /opt/google/compute-gpu-monitoring/linux
#sudo python3 -m venv venv
#sudo venv/bin/pip install wheel
#sudo venv/bin/pip install -Ur requirements.txt
#sudo cp /opt/google/compute-gpu-monitoring/linux/systemd/google_gpu_monitoring_agent_venv.service /lib/systemd/system
#sudo systemctl daemon-reload
#sudo systemctl --no-reload --now enable /lib/systemd/system/google_gpu_monitoring_agent_venv.service

# prepare FL
cd /home/prz_jab98
gsutil cp gs://fl-msc-segmentation-dataset/chest_dataset.zip .
unzip chest_dataset.zip

git clone -b cloud https://${token}@github.com/SanoScience/FederatedLearning_MSc.git
sudo chmod -R 777 FederatedLearning_MSc
cd FederatedLearning_MSc/segmentation


echo 'Running FL'
echo "export PYTHONPATH=$(dirname $PWD) && python3 client_segmentation.py ${address} ${index} ${node_count}" > run.sh
export PYTHONPATH=$(dirname $PWD)
#python3 client_segmentation.py ${address} ${index} ${node_count}
touch post.txt
