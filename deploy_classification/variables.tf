variable "total_node_count" {
  default = 10
}

variable "total_a100_node_count" {
  default = 6
}

variable "total_v100_node_count" {
  default = 4
}

variable "rounds" {
  default = 20
}

variable "local_epochs" {
  default = 2
}

variable "batch_size" {
  default = 64
}

variable "model" {
  default = "DenseNet121"
}

variable "training_datasets" {
  default = "mimic,chexpert,nih,chestdx"
}

variable "test_datasets" {
  default = "mimic,chexpert,nih,chestdx,chestdx-pe"
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 0.1
}

variable "min_fit_clients" {
  default = 10
}

variable "data_selection" {
  default = "iid"
}

variable "token" {
  type = string
}

variable "results_bucket" {
  default = "fl-msc-classification"
}

variable "study_prefix" {
  default = "mimic-chexpert-nih-chestdx"
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
    "chexpert",
    "chexpert",
    "chexpert",
    "mimic",
    "mimic",
    "mimic"]
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
