cmake_minimum_required(VERSION 3.21)
project(Orion)

set(CMAKE_CXX_STANDARD 17)

#add_compile_definitions("ORION_FLOAT") # use float instead of double for matrix calculations
#add_compile_definitions("ORION_NO_DEBUG") # uncomment for release builds
#add_compile_definitions("EIGEN_NO_DEBUG") # uncomment for release builds
#add_compile_definitions("ORION_COLMAJOR")

# ignore warning for virtual functions returning nothing
set(CMAKE_CXX_FLAGS "-O2 -Wno-return-type -Wreturn-local-addr -static")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(orion)
add_subdirectory(examples)

include_directories(lib/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME})