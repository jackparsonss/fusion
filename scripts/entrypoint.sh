#! /bin/bash

source ~/.bashrc

cd ../

mkdir build
cd build
cmake ..
make

echo "DIRECTORY"
pwd

cd ../tests
tester ./ci_config.json
