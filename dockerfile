FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install build-essential -y

RUN apt-get update -y
RUN apt-get install pkg-config -y

RUN apt-get update
RUN apt-get install uuid-dev -y

RUN apt-get update
RUN apt-get install openjdk-17-jre -y

RUN apt-get update
RUN apt-get install git -y

RUN apt-get update
RUN apt-get install cmake -y

RUN apt-get update
RUN apt-get install ninja-build -y

RUN apt-get update
RUN apt-get install curl -y

RUN mkdir /antlr
RUN cd /antlr && git clone https://github.com/antlr/antlr4.git
RUN cd /antlr/antlr4 && git checkout 4.13.1

RUN mkdir /antlr/antlr4/build
RUN mkdir /antlr/install
RUN cd /antlr/antlr4/build && cmake /antlr/antlr4/runtime/Cpp/ \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX="/antlr/install" && make install -j 8


RUN mkdir /antlr/install/bin
RUN curl https://www.antlr.org/download/antlr-4.13.1-complete.jar \
    -o /antlr/install/bin/antlr.jar

RUN cd / && git clone https://github.com/llvm/llvm-project.git
RUN cd /llvm-project && git checkout llvmorg-16.0.6
RUN mkdir /llvm-project/build
RUN cd /llvm-project/build && cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

RUN cd /llvm-project/build && ninja check-all -j 8 || true

RUN cd / && git clone https://github.com/cmput415/Tester.git
RUN mkdir /Tester/build
RUN cd /Tester/build && cmake .. && make -j 8

RUN echo 'export ANTLR_INS="/antlr/install"' >> ~/.bashrc
RUN echo 'export ANTLR_JAR="/antlr/install/bin/antlr.jar"' >> ~/.bashrc
RUN echo 'export CLASSPATH="$ANTLR_JAR:$CLASSPATH"' >> ~/.bashrc
RUN echo 'alias antlr4="java -Xmx500M org.antlr.v4.Tool"' >> ~/.bashrc
RUN echo 'alias grun="java org.antlr.v4.gui.TestRig"' >> ~/.bashrc

RUN echo 'export MLIR_INS="/llvm-project/build/"' >> ~/.bashrc
RUN echo 'export MLIR_DIR="/llvm-project/build/lib/cmake/mlir/"' >> ~/.bashrc
RUN echo 'export PATH="/llvm-project/build/bin:$PATH"' >> ~/.bashrc

RUN echo 'export PATH="/Tester/bin/:$PATH"' >> ~/.bashrc

RUN rm -rf /llvm-project/bolt
RUN rm -rf /llvm-project/clang
RUN rm -rf /llvm-project/clang-tools-extra
RUN rm -rf /llvm-project/compiler-rt
RUN rm /llvm-project/CONTRIBUTING.md
RUN rm -rf /llvm-project/cross-project-tests
RUN rm -rf /llvm-project/flang
RUN rm -rf /llvm-project/libc
RUN rm -rf /llvm-project/libclc
RUN rm -rf /llvm-project/libcxx
RUN rm -rf /llvm-project/libcxxabi
RUN rm -rf /llvm-project/libunwind
RUN rm /llvm-project/LICENSE.TXT
RUN rm /llvm-project/SECURITY.md
RUN rm -rf /llvm-project/lld
RUN rm -rf /llvm-project/lldb
RUN rm -rf /llvm-project/openmp
RUN rm -rf /llvm-project/polly
RUN rm -rf /llvm-project/pstl
RUN rm -rf /llvm-project/third-party
