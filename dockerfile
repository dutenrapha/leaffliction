FROM --platform=linux/amd64  ubuntu:24.04
# Platform amd64 necessario apenas para r-studio. Todo resto funciona sem isso

#COPY . /

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    git \
    vim \
    wget \
    zlib1g-dev \
    libncurses-dev \
    libgmp-dev \
    libreadline-dev \
    libssl-dev \
    libedit-dev \
    libunwind-dev \
    lsb-release \
    gnuplot \
    gnuplot-x11

# Install python system-wide
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-matplotlib

COPY ./requirements.txt /

RUN python3 -m pip install --break-system-packages -r requirements.txt

# Set default C compiler to clang
ENV CC=/usr/bin/clang

EXPOSE 8787

# Utilies
RUN apt-get install -y gnumeric
RUN python3 -m pip install visidata --break-system-packages


# Set working directory
WORKDIR /work

# Expose ports for development
EXPOSE 3000
EXPOSE 4000

# Set default command
CMD ["bash"]
