# FederatedLearning_MSc

Federated learning is a promising approach for secure distributed training of deep learning models, beginning to be applied in the healthcare sector, especially in medical image analysis. The goal of this research is to evaluate the impact of federated learning parameters and various data distributions on the training process of classification models, in addition to the analysis of hardware costs, using the selected largest publicly available chest X-ray datasets. All experiments are performed with high-performance computing resources and Google Cloud Platform.  Additionally, the datasets and the implementation of the designed federated learning system based on the Flower framework are described. We obtain average AUROC score on downsampled ChestX-ray14 test dataset between 0.801:0.817 for all configurations on ResNet-50 and DenseNet-121 models. We observe a speedup of about $3\times$ in federated training with 4 nodes compared to training on a single node. Furthermore, experiments combining multi-site data with over 600,000 chest X-ray images are conducted, showing an improvement in model performance compared to the use of data from one center. We achieve the most generalizable model with highest macro average AUROC of 0.864 in federated training using 4 distinct datasets.
