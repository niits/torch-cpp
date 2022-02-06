cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

add_executable(main src/main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 14)