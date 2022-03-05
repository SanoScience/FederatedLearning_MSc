variable "node_count" {
  default = 2
 }

variable "rounds" {
  default = 10
}

variable "fed_algo" {
  default = "FedAvg"
}

variable "optimizer" {
  default = "Adam"
}

variable "batch_size" {
  default = 2
}

variable "local_epochs" {
  default = 2
}

variable "learning_rate" {
  default = 0.0001
}

variable "fraction_fit" {
  default = 1.0
}

variable "min_fit_clients" {
  default = 2
}

variable "token" {
  type = string
}

variable "dataset_bucket" {
  default = "fl-msc-segmentation-dataset"

}