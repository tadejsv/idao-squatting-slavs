ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.1
FROM mambaorg/micromamba AS conda

COPY env.yml /
COPY .devcontainer/env_dev.yml /
RUN micromamba create -n idao && micromamba activate idao && \
    micromamba install -y -n idao -f /env.yml && \
    micromamba install -y -n idao -f /env_dev.yml && \
    pip install timm && \
    micromamba clean

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
COPY --from=conda /bin/micromamba /bin/micromamba
COPY --from=conda --chown=$USERNAME /opt/ /opt/
RUN sudo ln -s /opt/conda/etc/profile.d/mamba.sh /etc/profile.d/mamba.sh && \
    echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc && \
    echo "micromamba activate idao" >> ~/.bashrc
ENV MAMBA_EXE="/bin/micromamba"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Jupyter config
COPY --chown=$USERNAME .devcontainer/jupyter_lab_config.py $HOME/.jupyter/ 

# Prepare entrypoint and mount folder
WORKDIR /ws