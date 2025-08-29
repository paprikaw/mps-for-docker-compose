# MPS Control Guide

This repository provides two Compose setups for:
- Starting and managing the NVIDIA CUDA MPS control daemon (mps-control-daemon).
- Running inference/compute containers that connect to that MPS (e.g., a vLLM cluster).

Typical workflow: start the MPS daemon first, then launch workload containers to enable GPU sharing and concurrency.

Files:
- `docker-compose.yaml`: Starts the MPS control daemon on the host.
- `container-docker-compose.yaml`: Sample compose file for running containers configured with mps.

## Prerequisites
- NVIDIA driver installed on the host, `nvidia-smi` works.
- Docker/Podman on the host and NVIDIA container runtime (e.g., `nvidia-container-toolkit`).
- `container-docker-compose.yaml` is using podman specific syntax to require gpu resources, please follow the exact [steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) to configure Container Devices Interface and enable this for podman. 
- It is recommended to use podman and podman-compose to run these two compose file.
- Ensure these host directories exist (for MPS pipes and logs):
  - `/tmp/nvidia-mps`
  - `/var/log/nvidia-mps`

## Compose File Overview

### 1) `docker-compose.yaml` (MPS daemon)

- Key service: `services.mps-control-daemon`
  - `image: ubuntu:22.04`: Minimal userland image; the container will use the host’s NVIDIA driver and MPS binaries via chroot.
  - `privileged: true` and `pid: host`:
    - Grants enough privileges and shares host PID namespace to manage the MPS daemon and access host resources.
  - `entrypoint`: Executes a shell script after `chroot /driver-root`. Core logic:
    - Trap exit signals; on `TERM/INT`, send `quit` to `nvidia-cuda-mps-control` for graceful shutdown.
    - Run `nvidia-smi` to verify GPU and driver visibility.
    - Clean old startup marker: `/var/log/nvidia-mps/startup.log`.
    - Start MPS in background: `nvidia-cuda-mps-control -d`.
    - Optional settings:
      - Default active thread percentage: `set_default_active_thread_percentage ${MPS_ACTIVE_THREAD_PERCENTAGE}`.
      - Per-GPU pinned memory limit: `set_default_device_pinned_mem_limit <gpu_id> <limit>`.
    - Write a "startup complete" marker and tail MPS logs `control.log/server.log`.
  - `environment`:
    - `CUDA_VISIBLE_DEVICES`: GPU IDs visible to MPS (e.g., `"0"` or `"0,1"`).
    - `MPS_ACTIVE_THREAD_PERCENTAGE`: Default active thread percentage (0–100).
    - `MPS_PINNED_DEVICE_MEMORY_LIMITS`: Default Per-GPU pinned memory limits.
      - Note: The entrypoint script parses as `id:limit` (colon), e.g., `"0:10GB 1:8GB"`.
      - The current example uses `"0=10GB"` which doesn’t match. Prefer colon: `"0:10GB"`.

### 2) `container-docker-compose.yaml` (workload container cluster)

  This is a sample docker-compose file for running containers and configure MPS. In this file, we define two containers with bridge container network. It specify gpu memory and gpu cores percentages for each container individually. 

## Usage

To start mps daemon:
  - `podman compose -f docker-compose.yaml up`