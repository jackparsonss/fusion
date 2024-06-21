#! /bin/bash

source ~/.bashrc

cd ../

mkdir build
cd build
cmake ..
make

cd ../tests
tester ./tester_config.json
