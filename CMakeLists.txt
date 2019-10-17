cmake_minimum_required(VERSION 3.5)
project( Project-Janvier CXX )

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SRCS src/image.cpp src/coordinates.cpp src/main.cpp)
set(HEADERS include/image.h include/coordinates.h)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
include_directories(include)

add_executable(exe ${SRCS} ${HEADERS})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/exe
  DESTINATION ../bin)

target_compile_options( exe PUBLIC -std=c++11 -fpermissive -w -Wall )

target_link_libraries( exe ${OpenCV_LIBS} Eigen3::Eigen -L/usr/lib64 -ldl )
