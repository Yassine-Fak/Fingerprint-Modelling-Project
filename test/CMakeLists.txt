cmake_minimum_required(VERSION 3.5)
project( Project-Janvier CXX )

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

set(SRCS ./tests.cpp ./image.cpp ./coordinates.cpp ./pixels.cpp)
set(HEADERS ./image.h ./coordinates.h ./pixels.h)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY .)
include_directories(.)

add_executable(exe ${SRCS} ${HEADERS})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/exe
  DESTINATION .)

target_compile_options( exe PUBLIC -std=c++11 -fpermissive -w -Wall )

target_link_libraries( exe ${OpenCV_LIBS} Eigen3::Eigen GTest -L/usr/lib64 -ldl )
