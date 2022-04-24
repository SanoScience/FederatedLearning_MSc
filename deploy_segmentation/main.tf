provider "google" {
  project = "sano-332607"
  region = "us-central1"
  zone = "us-central1-a"
}

resource google_service_account "default" {
  account_id = "flower-federated-learning"
  display_name = "FL Msc Account"
}

resource "google_storage_bucket_iam_binding" "binding" {
  bucket = var.dataset_bucket
  role = "roles/storage.admin"
  members = [
    "serviceAccount:${google_service_account.default.email}"
  ]
}

module "flower-vpc" {
  source = "terraform-google-modules/network/google"
  project_id = "sano-332607"
  network_name = "flower-server-vpc"
  routing_mode = "GLOBAL"
  auto_create_subnetworks = true

  subnets = [
    {
      subnet_name = "flower-subnet"
      subnet_ip = "10.10.10.0/24"
      subnet_region = "us-central1"
    }]

  routes = [
    {
      name = "egress2-internet"
      description = "route through IGW to access internet"
      destination_range = "0.0.0.0/0"
      tags = "egress2-inet"
      next_hop_internet = "true"
    }]
}

resource google_compute_firewall "firewall-server" {
  name = "firewall-server"
  network = "default"
  source_ranges = [
    "0.0.0.0/0"]
  // todo: use serivce account
  // source_service_accounts = [google_service_account.default.email]
  allow {
    protocol = "tcp"
    ports = [
      "80",
      "443",
      "5000-5999",
      "6000-6999",
      "7000-7999",
      "8000-8999"]
  }
}

resource "google_compute_address" "flower-server" {
  name = "flower-server"
}

resource "google_compute_instance" "server" {
  name = "server"
  machine_type = "n1-standard-8"
  zone = "us-central1-a"
  tags = [
    "flwr-server"]

  boot_disk {
    initialize_params {
      image = "projects/sano-332607/global/images/fl-msc-image-v1"
    }
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = google_compute_address.flower-server.address
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-k80"
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
  metadata_startup_script = file("./server_startup.sh")

}


resource google_compute_instance "client" {
  name = "client-${count.index}"
  machine_type = "n1-standard-8"
  zone = "us-central1-a"
  count = var.node_count
  tags = [
    "flwr-client"]
  boot_disk {
    initialize_params {
      image = "projects/sano-332607/global/images/fl-msc-image-v1"
    }
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-k80"
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

  metadata_startup_script = file("./client_startup.sh")
}

output "instance_0_endpoint" {
  value = google_compute_address.flower-server.address
}