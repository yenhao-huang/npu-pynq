# Linux x86-64 environment for AMD Vivado on Apple silicon (Docker Desktop +
# Rosetta). Vivado itself is not in the image: it is installed once, from the
# user's AMD installer, into the volume mounted at /tools/Xilinx.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales ca-certificates sudo git make python3 python3-pip \
        libtinfo5 libncurses5 libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
        libfreetype6 fontconfig fonts-dejavu-core libgtk2.0-0 libgtk-3-0 \
        libglib2.0-0 libnss3 libasound2 libxft2 libxss1 libcanberra-gtk-module \
        libsecret-1-0 graphviz unzip xz-utils \
        tigervnc-standalone-server tigervnc-common openbox xterm x11-xserver-utils \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Same user name as the board-side scripts expect nothing of; uid 1000 keeps
# files in the mounted repository writable.
RUN useradd -m -u 1000 -s /bin/bash vivado && echo 'vivado ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/vivado \
    && mkdir -p /tools/Xilinx && chown vivado /tools/Xilinx
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
USER vivado
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
