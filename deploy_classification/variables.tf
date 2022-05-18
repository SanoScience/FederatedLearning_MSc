variable "total_node_count" {
  default = 4
}

variable "total_a100_node_count" {
  default = 4
}

variable "total_v100_node_count" {
  default = 0
}

variable "rounds" {
  default = 100
}

variable "local_epochs" {
  default = 1
}

variable "batch_size" {
  default = 128
}

variable "model" {
  default = "DenseNet121"
}

variable "training_datasets" {
  default = "cc-cxri-p"
}

variable "test_datasets" {
  default = "cc-cxri-p"
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 0.1
}

variable "min_fit_clients" {
  default = 4
}

variable "data_selection" {
  default = "noniid"
}

variable "token" {
  type = string
}

variable "results_bucket" {
  default = "fl-msc-classification"
}

variable "study_prefix" {
  default = "cc-cxri-p-noniid"
}

variable "a100_client_zones" {
  default = [
    "asia-northeast1-a",
    "asia-southeast1-b",
    "europe-west4-a",
    "us-central1-a",
    "us-east1-b",
    "us-west1-b"]
}

variable "v100_client_zones" {
  default = [
    "asia-east1-c",
    "europe-west4-a",
    "us-east1-c",
    "us-west1-a"]
}

variable "a100_client_datasets" {
  default = [
    "cc-cxri-p",
    "cc-cxri-p",
    "cc-cxri-p",
    "cc-cxri-p"]
}

variable "v100_client_datasets" {
  default = [
    "chestdx",
    "chestdx",
    "nih",
    "nih"]
}

variable "a100_client_indices" {
  default = [
    0,
    1,
    2,
    0,
    1,
    2]
}

variable "v100_client_indices" {
  default = [
    0,
    1,
    0,
    1]
}

variable "a100_client_counts" {
  default = [
    3,
    3,
    3,
    3,
    3,
    3]
}

variable "v100_client_counts" {
  default = [
    2,
    2,
    2,
    2]
}
