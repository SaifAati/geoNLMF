cmake_minimum_required(VERSION 3.19)
project(geoNLMF)

set(CMAKE_CXX_STANDARD 14)

include_directories(Cpp/Include)

add_executable(geoNLMF
        Cpp/Include/geoNLMF.h
        Cpp/Include/geoNLMFLibTypeCode.h
        Cpp/geoNLMF.cpp
        Cpp/geoNLMFLib.cpp)
