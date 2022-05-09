variable "node_count" {
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
  default = "DenseNet121"
}

variable "training_datasets" {
  default = "nih"
}

variable "test_datasets" {
  default = "nih,chestdx,chestdx-pe"
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
  default = "nih_chestdx"
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
    "nih",
    "nih",
    "nih"]
}

variable "client_ids" {
  default = [
    0,
    1,
    2]
}
