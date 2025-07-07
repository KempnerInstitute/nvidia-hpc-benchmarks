# NVIDIA HPC Container to Singularity

NVIDIA provides a ready-to-use container for HPC benchmarks through its NGC (NVIDIA GPU Cloud) platform. However, this container is not directly compatible with Singularity, which is the standard container runtime in many HPC environments like the Kempner AI Cluster. To run these benchmarks, you need to convert the NGC container into a Singularity image file (`.sif`).

Below are clear steps to pull, convert, and prepare the container for benchmarking on your cluster.

### Step 1: Create and NGC account

You need an NGC account to access the NVIDIA HPC container registry:

  - https://ngc.nvidia.com/

> [!Note] 
> Check with the cluster administrators if they have already pulled the container images and made them available in a shared directory. If so, you can skip the steps below and use the provided `.sif` image directly.

### Step 2: Pull the container image with Podman

NVIDIA’s images follow Open Container Initiative (OCI) standards. In theory, Singularity can pull them directly.
However, for complex multi-layered containers, you may see conversion bugs like:

```bash
panic: runtime error: index out of range [9] with length 9
```
This is a known Singularity issue with deeply layered OCI images.
To avoid it, first pull the image with Podman, which handles OCI layers robustly.

Run this on a compute node (not the login node!):

```bash
podman pull nvcr.io/nvidia/hpc-benchmarks:25.04
```

> [!Note] 
> You may need to add username and API key to get the image. 


Check if you have the image locally:

```bash
podman images
```
This should show the `nvcr.io/nvidia/hpc-benchmarks:25.04` image in the list.

```bash
REPOSITORY                     TAG         IMAGE ID      CREATED       SIZE
nvcr.io/nvidia/hpc-benchmarks  25.04       e2a9e5d2d87f  3 months ago  12.5 GB
```
(Optional) If you want to keep a local copy of the image, you can save it as a tar file. This is useful for transferring the image to other systems or for backup purposes.

```bash
podman save nvcr.io/nvidia/hpc-benchmarks:25.04 -o hpc-benchmarks-25.04.tar
```

### Step 3: Start the Podman RESR API

Singularity’s build command needs Docker-compatible REST APIs. Podman provides this via its service:

```bash
podman system service --time=0 unix:///tmp/podman.sock &
```

### Step 4: Export the Podman socket to Docker

```bash
export DOCKER_HOST=unix:///tmp/podman.sock
```

### Step 5: Build the Singularity `.sif`image 

```bash
singularity build ./nvidia-hpc-benchmarks-25-04.sif docker-daemon://nvcr.io/nvidia/hpc-benchmarks:25.04
```

This creates a Singularity image file named `nvidia-hpc-benchmarks-25-04.sif` in your current directory. Here are the specs for the image:

| Name                            | size   | 
| ------------------------------- | ------ |
| nvidia-hpc-benchmarks-25-04.sif | 6.8 GB | 

### Step 6: Stop the Podman service

```bash
jobs     # find the job number
kill %1  # or the PID
```

### Done! 

You now have a Singularity image file (`.sif`) that contains the NVIDIA HPC benchmarks, ready to run on the Kempner AI Cluster.

> [!Note] 
> It is possible to directly pull the image from NGC and convert it to Singularity format, however, sometimes you may get runtime error (e.g., `panic: runtime error: index out of range [9] with length 9`). This happens because sometimes Singularity hits a bug when converting very large, multi-layer OCI images. Using Podman to pull the image first and then converting it to Singularity format usually resolves this issue, as Podman is compatible with OCI standards and can handle complex multi-layered images more gracefully.
