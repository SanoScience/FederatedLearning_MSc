provider "google" {
  project = "sano-332607"
  region = "us-central1"
  zone = "us-central1-a"
}

resource google_service_account "default" {
  account_id = "flower-federated-learning"
  display_name = "FL Msc Account for classification"
}

resource "google_storage_bucket_iam_binding" "binding" {
  bucket = var.results_bucket
  role = "roles/storage.admin"
  members = [
    "serviceAccount:${google_service_account.default.email}"
  ]
}

resource "google_project_iam_member" "monitoring-writer" {
  project = "sano-332607"
  role   = "roles/monitoring.metricWriter"
  member = "serviceAccount:${google_service_account.default.email}"
}

resource "google_project_iam_member" "logs-writer" {
  project = "sano-332607"
  role   = "roles/logging.logWriter"
  member = "serviceAccount:${google_service_account.default.email}"
}

//module "flower-classification-vpc" {
//  source = "terraform-google-modules/network/google"
//  project_id = "sano-332607"
//  network_name = "flower-classification-vpc"
//  routing_mode = "GLOBAL"
//  auto_create_subnetworks = true
//
//  subnets = [
//    {
//      subnet_name = "flower-classification-subnet"
//      subnet_ip = "10.10.10.0/24"
//      subnet_region = "us-central1"
//    }]
//
//  routes = [
//    {
//      name = "egress2-internet"
//      description = "route through IGW to access internet"
//      destination_range = "0.0.0.0/0"
//      tags = "egress2-inet"
//      next_hop_internet = "true"
//    }]
//}

resource google_compute_firewall "firewall-classification-server" {
  name = "firewall-classification-server"
  network = "default"
  source_ranges = [
    "0.0.0.0/0"]
  // todo: use serivce account
  // source_service_accounts = [google_service_account.default.email]
  allow {
    protocol = "tcp"
    ports = [
      "22",
      "80",
      "443",
      "5000-5999",
      "6000-6999",
      "7000-7999",
      "8000-8999"]
  }
}

resource "google_compute_address" "flower-classification-server" {
  name = "flower-classification-server"
}

resource "google_compute_instance" "server" {
  name = "classification-server"
  machine_type = "n1-standard-8"
  zone = "us-central1-a"
  tags = [
    "flwr-classification-server"]

  boot_disk {
    initialize_params {
      image = "projects/sano-332607/global/images/fl-classification-v1"
    }
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = google_compute_address.flower-classification-server.address
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-v100"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }
  service_account {
    email = google_service_account.default.email
    scopes = [
      "cloud-platform"]
  }
  metadata_startup_script = templatefile("./server_startup.sh", {
    token = var.token
    fraction_fit = var.fraction_fit
    min_fit_clients = var.min_fit_clients
    learning_rate = var.learning_rate
    local_epochs = var.local_epochs
    batch_size = var.batch_size
    rounds = var.rounds
    node_count = var.node_count
    model = var.model
    data_selection = var.data_selection
    training_datasets = var.training_datasets
    test_datasets = var.test_datasets
    results_bucket = var.results_bucket
    study_prefix = var.study_prefix
  })

}

resource google_compute_instance "client" {
  name = "classification-client-${count.index}"
  machine_type = "a2-highgpu-1g"
  count = var.node_count
  zone = var.a_100_client_zones[count.index]
  tags = [
    "flwr-classification-client"]
  boot_disk {
    initialize_params {
      image = "projects/sano-332607/global/images/fl-classification-v1"
    }
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

//  guest_accelerator {
//    type = "nvidia-tesla-a100"
//    count = 1
//  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  service_account {
    email = google_service_account.default.email
    scopes = [
      "cloud-platform"]
  }

  metadata_startup_script = templatefile("./client_startup.sh", {
    token = var.token
    address = google_compute_address.flower-classification-server.address
    index = count.index
    node_count = var.node_count
    client_dataset = var.client_datasets[count.index]
  })
}

output "instance_0_endpoint" {
  value = google_compute_address.flower-classification-server.address
}