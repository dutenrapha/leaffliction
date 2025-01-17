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
COPY ./requirements.txt /

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-matplotlib
# Precisa instalar matplotlib pelo apt-get senoa nao plota imagens
# de dentro do container

# Install SBCL (Steel Bank Common Lisp)
RUN apt-get install -y clisp cl-quicklisp

# Install SWI-Prolog
RUN apt-get install -y swi-prolog

# Install Clojure
RUN apt-get install -y clojure

# Set default C compiler to clang
ENV CC=/usr/bin/clang

# ========== R Instalation ================
# Instructions from:
# https://cloud.r-project.org/bin/linux/ubuntu/fullREADME.html

RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" >> /etc/apt/sources.list.d/ubuntu.sources
RUN apt-get update && apt-get install -y r-base

# ========== R Studio Instalation ================
# Instructions from:
# https://posit.co/download/rstudio-server/ 

# Se nao criar um usuario nao funciona.
RUN useradd rstudio && passwd -d rstudio && mkdir /home/rstudio && chown -R rstudio /home/rstudio

RUN apt-get install -y gdebi-core
RUN wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.04.2-764-amd64.deb
RUN gdebi --n rstudio-server-2024.04.2-764-amd64.deb
# Necessario para entrar no rstudio sem autenticacao
RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf
RUN rm rstudio-server-2024.04.2-764-amd64.deb

# Utilies
RUN apt-get install -y gnumeric
RUN python3 -m pip install visidata --break-system-packages
RUN cat requirements.txt | xargs -n 1 python3 -m pip install --break-system-packages

# Set working directory
WORKDIR /work

# Expose ports for development
EXPOSE 3000
EXPOSE 4000
EXPOSE 8787
EXPOSE 8888

# Set default command
CMD ["bash"]
