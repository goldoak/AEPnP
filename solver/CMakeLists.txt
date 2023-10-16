cmake_minimum_required(VERSION 3.10.0)
project(aPnP VERSION 0.1.0)

find_package(Eigen3 REQUIRED)
include_directories(/usr/include/eigen3)

set(SOURCE_FILE 
    ./src/anisotropicPnP.hpp
    ./src/anisotropicPnP.cpp
    ./src/apnp.hpp
)

add_subdirectory(extern/pybind11)
pybind11_add_module(apnp src/apnp.cpp ${SOURCE_FILE})
