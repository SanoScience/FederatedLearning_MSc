variable "node_count" {
  default = 4
 }

variable "rounds" {
  default = 12
}

variable "fed_algo" {
  default = "FedAdam"
}

variable "optimizer" {
  default = "Adam"
}

variable "batch_size" {
  default = 16
}

variable "local_epochs" {
  default = 3
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 1.0
}

variable "min_fit_clients" {
  default = 4
}

variable "token" {
  type = string
}

variable "dataset_bucket" {
  default = "fl-msc-segmentation-dataset"
}