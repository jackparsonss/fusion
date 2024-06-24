#! /bin/bash

source ~/.bashrc

cd ../

mkdir build
cd build
cmake ..
make

cd ../tests
tester ./ci_config.json
