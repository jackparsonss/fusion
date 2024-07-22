#! /bin/bash

# install xcode command line tools
if ! xcode-select -p &>/dev/null; then
    echo "Xcode command line tools not found. Installing Xcode command line tools..."
    xcode-select --install

    # Wait for the installation to complete
    until xcode-select -p &>/dev/null; do
        sleep 5
    done

    echo "Xcode command line tools installed successfully."
else
    echo "Xcode command line tools are already installed."
fi

# install homebrew
if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to the PATH
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >>~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "Homebrew is already installed."
fi

# install java
if ! command -v java &>/dev/null; then
    echo "Java not found. Installing Oracle Java JDK..."

    # Determine the architecture
    ARCH=$(uname -m)
    if [ "$ARCH" == "arm64" ]; then
        JDK_URL="https://download.oracle.com/java/21/latest/jdk-21_macos-aarch64_bin.tar.gz"
        JDK_TAR="jdk-21_macos-aarch64_bin.tar.gz"
        JDK_DIR="jdk-21.0.3.jdk"
    elif [ "$ARCH" == "x86_64" ]; then
        JDK_URL="https://download.oracle.com/java/21/latest/jdk-21_macos-x64_bin.tar.gz"
        JDK_TAR="jdk-21_macos-x64_bin.tar.gz"
        JDK_DIR="jdk-21.0.3.jdk"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi

    # Download the JDK tarball
    curl -LO "$JDK_URL"

    # Extract the JDK tarball
    tar -xzf "$JDK_TAR"

    # Move the JDK to /opt (or any preferred location)
    sudo mv "$JDK_DIR" /opt/jdk-21

    # Set up the JAVA_HOME environment variable and update PATH
    echo 'export JAVA_HOME=/opt/jdk-21/Contents/Home' >>~/.zshrc
    echo 'export PATH=$JAVA_HOME/bin:$PATH' >>~/.zshrc

    # Source the profile to update the current session
    source ~/.zshrc

    rm ./jdk-21_macos-aarch64_bin.tar.gz

    echo "Oracle Java JDK installed successfully."
else
    echo "Java is already installed."
fi

# install cmake
if ! command -v cmake &>/dev/null; then
    echo "CMake not found. Installing CMake..."

    # Install CMake using Homebrew
    brew install cmake

    echo "CMake installed successfully."
else
    echo "CMake is already installed."
fi

# install ANTLR4 runtime
ANTLR_PARENT=$HOME/antlr
ANTLR_BIN=$HOME/antlr/antlr4-install/bin
SRC_DIR=$HOME/antlr/antlr4
BUILD_DIR=$HOME/antlr/antlr4/antlr4-build
INSTALL_DIR=$HOME/antlr/antlr4-install
THREADS=$(sysctl -n hw.ncpu)

if [ ! -d "$ANTLR_PARENT" ]; then
    mkdir -p "$ANTLR_PARENT"
    echo "Directory $ANTLR_PARENT created."

    cd "$ANTLR_PARENT"
    git clone https://github.com/antlr/antlr4.git ~/antlr/antlr4

    cd "$HOME/antlr/antlr4"
    git checkout 4.13.1

    mkdir -p "$BUILD_DIR"
    echo "Directory $BUILD_DIR created."

    mkdir -p "$INSTALL_DIR"
    echo "Directory $INSTALL_DIR created."

    cd "$BUILD_DIR"
    echo "building ANTLR4..."
    cmake "$SRC_DIR"/runtime/Cpp/ \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
    make install -j $THREADS

    echo "export ANTLR_INS=\"$INSTALL_DIR\"" >>~/.zshrc

    echo "installing ANTLR4 generator"
    mkdir -p "$ANTLR_BIN"

    curl https://www.antlr.org/download/antlr-4.13.1-complete.jar -o "$ANTLR_BIN"/antlr-4.13.1-complete.jar

    ANTLR_JAR="$ANTLR_BIN/antlr-4.13.1-complete.jar"
    CLASSPATH="$ANTLR_JAR:$CLASSPATH"
    ANTLR_ALIAS="java -Xmx500M org.antlr.v4.Tool"
    GRUN_ALIAS="java org.antlr.v4.gui.TestRig"
    echo "export ANTLR_JAR=\"$ANTLR_JAR\"" >>~/.zshrc
    echo "export CLASSPATH=\"$CLASSPATH\"" >>~/.zshrc
    echo "alias antlr4=\"$ANTLR_ALIAS\"" >>~/.zshrc
    echo "alias grun=\"$GRUN_ALIAS\"" >>~/.zshrc

    source ~/.zshrc

else
    echo "ANLTR4 already installed"
fi

# installing mlir
export MLIR_INS="$HOME/llvm-project/build"
export MLIR_DIR="$MLIR_INS/lib/cmake/mlir"

if [ ! -d "$HOME/llvm-project" ]; then
    git clone https://github.com/llvm/llvm-project.git ~/llvm-project
    cd "$HOME/llvm-project"
    git checkout llvmorg-18.1.8

    brew install ninja

    mkdir -p "$MLIR_INS"
    mkdir -p "$MLIR_INS"

    cmake -G Ninja ../llvm-project \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON

    ninja check-all -j $THREADS

    PATH="$MLIR_INS/bin:'$PATH'"
    echo "export MLIR_INS=\"$MLIR_INS\"" >>~/.zshrc
    echo "export MLIR_DIR=\"$MLIR_DIR\"" >>~/.zshrc
    echo "export PATH=\"$PATH\"" >>~/.zshrc

    source ~/.zshrc
else
    echo "MLIR already installed"
fi
