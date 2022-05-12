variable "total_node_count" {
  default = 3
}

variable "rounds" {
  default = 20
}

variable "local_epochs" {
  default = 2
}

variable "batch_size" {
  default = 128
}

variable "model" {
  default = "ResNet50"
}

variable "training_datasets" {
  default = "mimic"
}

variable "test_datasets" {
  default = "mimic,chexpert"
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 0.1
}

variable "min_fit_clients" {
  default = 3
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
  default = "mimic-chexpert"
}

variable "a_100_client_zones" {
  default = [
    "asia-northeast1-a",
    "asia-southeast1-b",
    "europe-west4-a",
    "us-central1-a",
    "us-east1-b",
    "us-west1-b"]
}

variable "client_datasets" {
  default = [
    "mimic",
    "mimic",
    "mimic"]
}

variable "client_indices" {
  default = [
    0,
    1,
    2]
}

variable "client_counts" {
  default = [
    3,
    3,
    3]
}
