cmake_minimum_required(VERSION 2.8.3)
project(net)

add_compile_options(-std=c++11 -g)

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
	${OpenCV_LIBRARY_DIRS}
)

add_executable(net src/simple.cc
)
