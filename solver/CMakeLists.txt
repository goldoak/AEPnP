cmake_minimum_required(VERSION 3.10.0)
project(aEPnP VERSION 0.1.0)

find_package(Eigen3 REQUIRED)
include_directories(/usr/include/eigen3)

set(SOURCE_FILE
	./src/AnisotropicEpnp.hpp
	./src/AnisotropicEpnp.cpp
	./src/aepnp.hpp)

add_subdirectory(extern/pybind11)
pybind11_add_module(aepnp src/aepnp.cpp ${SOURCE_FILE})
