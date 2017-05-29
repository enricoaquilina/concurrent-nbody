cmake_minimum_required(VERSION 3.6)
project(nbodyAss)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES main.cpp common.h)
add_executable(nbodyAss ${SOURCE_FILES})