FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV ANTLR_PARENT=/root/antlr
ENV SRC_DIR=/root/antlr/antlr4
ENV BUILD_DIR=/root/antlr/antlr4/antlr-build
ENV INSTALL_DIR=/root/antlr/antlr4-install
ENV ANTLR_BIN=/root/antlr/antlr4-install/bin
ENV ANTLR_INS=/root/antlr/antlr4-install
ENV ANTLR_JAR=/root/antlr/antlr4-install/bin/antlr-4.13.1-complete.jar


RUN apt-get update
RUN apt-get install -y \
    git \
    cmake \
    neovim \
    ncal \
    build-essential \
    pkg-config \
    uuid-dev \
    openjdk-17-jre \
    cmake \
    curl \
    wget \
    ninja-build

# Build antlr
RUN mkdir -p ${INSTALL_DIR}
WORKDIR /root/antlr
RUN git clone https://github.com/antlr/antlr4.git
WORKDIR /root/antlr/antlr4
RUN git checkout 4.13.1
WORKDIR /root/antlr/antlr4/antlr4-build
RUN cmake ${SRC_DIR}/runtime/Cpp/ -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
RUN make install -j 8
RUN echo "export ANTLR_INS=/root/antlr/antlr4-install" >> /root/.bashrc

# Antlr generator
RUN mkdir -p ${ANTLR_BIN}
RUN curl https://www.antlr.org/download/antlr-4.13.1-complete.jar -o ${ANTLR_BIN}/antlr-4.13.1-complete.jar

RUN echo 'export ANTLR_JAR="/root/antlr/antlr4-install/bin/antlr-4.13.1-complete.jar"' >> /root/.bashrc
RUN echo 'export CLASSPATH="$ANTLR_JAR:$CLASSPATH"' >> /root/.bashrc
RUN echo 'alias antlr4="java -Xmx500M org.antlr.v4.Tool"' >> /root/.bashrc
RUN echo 'alias grun="java org.antlr.v4.gui.TestRig"' >> /root/.bashrc

# Installing the tester
WORKDIR /root
RUN git clone https://github.com/cmput415/Tester.git
WORKDIR /root/Tester
RUN mkdir build
WORKDIR /root/Tester/build
RUN cmake ..
RUN make

RUN echo 'export PATH="/root/Tester/bin/:$PATH"' >> /root/.bashrc

ENV CLASSPATH="$ANTLR_JAR:$CLASSPATH"
ENV MLIR_INS=/root/llvm-project/build
ENV MLIR_DIR=/root/llvm-project/build/lib/cmake/mlir
ENV PATH="/root/llvm-project/build/bin:/root/Tester/bin:$PATH"


# Installing mlir
WORKDIR /root
RUN git clone https://github.com/llvm/llvm-project.git
WORKDIR llvm-project
RUN git checkout llvmorg-18.1.8
RUN mkdir build

WORKDIR build
RUN cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

RUN ninja -j8

RUN echo 'export MLIR_INS="$HOME/llvm-project/build"' >> /root/.bashrc
RUN echo 'export MLIR_DIR="${MLIR_INS}lib/cmake/mlir"' >> /root/.bashrc
RUN echo 'export PATH="$MLIR_INS/bin:$PATH"' >> /root/.bashrc


# Start in the root directory
WORKDIR /root

CMD ["/usr/bin/bash", "--login"]

