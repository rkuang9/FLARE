cmake_minimum_required(VERSION 3.21)
project(Orion)

set(CMAKE_CXX_STANDARD 17)

#add_compile_definitions("ORION_FLOAT") # use float instead of double for matrix calculations
#add_compile_definitions("ORION_NO_DEBUG") # uncomment for release builds
#add_compile_definitions("EIGEN_NO_DEBUG") # uncomment for release builds
#add_compile_definitions("EIGEN_DEFAULT_TO_ROW_MAJOR")


set(CMAKE_CXX_FLAGS "-O0  -Wno-return-type -Wreturn-local-addr -static")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(orion)

include_directories(lib/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(lib/Boost_1_78_0) # for JSON parsing

add_executable(${PROJECT_NAME} ${SRC_FILES} orion/weights/random_generator.cpp)

target_link_libraries(${PROJECT_NAME})