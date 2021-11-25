cmake_minimum_required(VERSION 3.1)

set (EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -O3 -Wall -fopenmp -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "-lm -pthread -lpthread")
set(CMAKE_BUILD_TYPE "Release")


project(geoNLMF)

message("-----------------------------------------------------------")
message("Setting CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
message(STATUS "${PROJECT_SOURCE_DIR}")
message("-----------------------------------------------------------")

set (LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
# find_path(INCLUDE_DIR PATHS ${PROJECT_SOURCE_DIR}/include)