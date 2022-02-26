provider "google" {
    project = "sano-332607"
    region = "us-central1"
    zone = "us-central1-a"
  }


module "flower-vpc" {
  source = "terraform-google-modules/network/google"
  project_id = "sano-332607"
  network_name = "flower-server-vpc"
  routing_mode = "GLOBAL"
  auto_create_subnetworks = true

  subnets = {
    subnet_name = "flower-subnet"
    subnet_ip = "10.10.10.0/24"
    subnet_region = "us-central1"
  }

  routes = {
    name = "egress2-internet"
    description = "route through IGW to access internet"
    destination_range = "0.0.0.0/0"
    tags = "egress2-inet"
    next_hop_internet = "true"
  }
}

resource google_compute_firewall "firewall-server" {
  name = "firewall-server"
  network = "default"
  source_tags = ["web"]

  allow = {
    protocol = "tcp"
    ports = ["80", "8080", "443", "5000-5999", "6000-6999", "7000-7999"]
  }
}

resource "google_compute_address" "flower-server" {
  name = "flower-server"
}

resource "google_compute_instance" "server" {
  name = "server"
  machine_type = "n1-standard-8"
  zone = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "projects/sano-332607/global/images/fl-msc-image-v1"
    }
  }

  network_interface = {
    network = "default"

    access_config = {
      nat_ip = google_compute_address.flower-server.address
    }
  }

  guest_accelerator = {
    type = "nvidia-tesla-k80"
    count = "1"
  }

  scheduling = {
    on_host_maintenance = "TERMINATE"
  }

  metadata_startup_script = "#!/bin/bash\n\necho hello"
}

output "instance_0_endpoint" {
  value = google_compute_address.flower-server.address
}