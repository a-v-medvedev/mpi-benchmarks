#!/bin/bash

set -e 

ARGSPARSER_VERSION=0.0.13
YAML_VERSION=0.7.0

function download() {
    [ ! -f yaml-cpp-$YAML_VERSION.tar.gz ] && wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-$YAML_VERSION.tar.gz
    [ ! -f v$ARGSPARSER_VERSION.tar.gz ] && wget https://github.com/a-v-medvedev/argsparser/archive/v$ARGSPARSER_VERSION.tar.gz
    true
}

function unpack() {
    [ -e yaml-cpp-$YAML_VERSION.tar.gz -a ! -e yaml-cpp-yaml-cpp-$YAML_VERSION ] && tar xzf yaml-cpp-$YAML_VERSION.tar.gz
    [ -e v$ARGSPARSER_VERSION.tar.gz -a ! -e argsparser-$ARGSPARSER_VERSION ] && tar xzf v$ARGSPARSER_VERSION.tar.gz
    cd argsparser-$ARGSPARSER_VERSION
    [ ! -e yaml-cpp -a ! -L yaml-cpp ] && ln -s ../yaml-cpp yaml-cpp 
    cd ..
}

function build() {
    cd yaml-cpp-yaml-cpp-$YAML_VERSION
    [ -e build ] && rm -rf build
    mkdir -p build
    cd build
    cmake -DBUILD_SHARED_LIBS=ON -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF .. -DCMAKE_INSTALL_PREFIX=$PWD/../../yaml-cpp
    make clean
    make -j
    make install
    cd ../..
    [ -d yaml-cpp/lib64 -a ! -e yaml-cpp/lib ] && ln -s lib64 yaml-cpp/lib

    cd argsparser-$ARGSPARSER_VERSION && make && cd ..
}

function install() {
    mkdir -p include
    mkdir -p lib
    cp -v argsparser-$ARGSPARSER_VERSION/argsparser.h include/
    cp -v argsparser-$ARGSPARSER_VERSION/libargsparser.so lib/
    cp -rv argsparser-$ARGSPARSER_VERSION/extensions include
    cp -av yaml-cpp/include/yaml-cpp include/ 
    cp -av yaml-cpp/lib/* lib/
}

download
unpack
build
install
