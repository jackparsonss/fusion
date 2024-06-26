#! /bin/bash

source ~/.bashrc

cd ../

mkdir build
cd build
cmake ..
make -j 5

scan-build make

cd ../tests
tester ./ci_config.json
