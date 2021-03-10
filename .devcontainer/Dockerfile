ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.1
FROM mambaforge/micromamba AS conda

COPY env.yml /root
COPY .devcontainer/env_dev.yml /root
RUN micromamba install -y -n base -f /root/env.yaml && \
    micromamba install -y -n base -f /root/env_dev.yaml && \
    rm /opt/conda/pkgs/cache/*

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# Prepare shell and file system
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 SHELL=/bin/bash
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install all system stuff, including node
COPY sys_requirements.txt /tmp
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt install -y --no-install-recommends \
    $(cat tmp/sys_requirements.txt) && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create the sudo user
ARG USERNAME
ARG UID
RUN useradd $USERNAME -u $UID -G sudo -s /bin/bash -m && \
    echo $USERNAME' ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME
ENV HOME=/home/$USERNAME
RUN sudo mkdir /ws && sudo chown -R $USERNAME /ws

# Copy over conda and bashrc, install environment
COPY --from=conda --chown=$USERNAME /opt/ /opt/
COPY --from=conda --chown=$USERNAME /bin/micromamba /bin/micromamba
RUN /bin/micromamba shell init -s bash -p /opt/conda && \
    echo "micromamba activate base" >> /root/.bashrc

# Jupyter config
COPY --chown=$USERNAME .devcontainer/jupyter_lab_config.py $HOME/.jupyter/ 

# Prepare entrypoint and mount folder
WORKDIR /ws