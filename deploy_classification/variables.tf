variable "node_count" {
  default = 8
}

variable "rounds" {
  default = 30
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
  default = "nih"
}

variable "test_datasets" {
  default = "nih,chestdx,chestdx-pe"
}

variable "client_dataset" {
  default = "nih"
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 0.1
}

variable "min_fit_clients" {
  default = 8
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
