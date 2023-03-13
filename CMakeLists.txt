cmake_minimum_required(VERSION 3.21)
project(FLARE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -openmp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#add_compile_definitions("FLARE_FLOAT") # use float instead of double for tensor calculations
#add_compile_definitions("FLARE_NO_DEBUG") # uncomment for release builds

add_compile_definitions("EIGEN_USE_THREADS")
add_compile_definitions("EIGEN_DEFAULT_TO_ROW_MAJOR")

#add_compile_definitions("USING_OPENCV") # include OpenCV for image-to-tensor


# without this, nothing works
include_directories(lib/Eigen)

get_directory_property(DEFINITIONS COMPILE_DEFINITIONS)
if (DEFINITIONS MATCHES "USING_OPENCV")
    set(OpenCV_DIR lib/opencv)
    message("INCLUDING OPENCV AT PATH: " ${OpenCV_DIR})
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message("THIS BUILD DOES NOT INCLUDE OpenCV FOR IMAGE PARSING")
endif ()

message("BUILD TYPE: " ${CMAKE_BUILD_TYPE})


add_subdirectory(flare)
add_subdirectory(examples)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})