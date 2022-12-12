cmake_minimum_required(VERSION 3.21)
project(Orion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -openmp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#add_compile_definitions("ORION_FLOAT") # use float instead of double for tensor calculations
#add_compile_definitions("ORION_NO_DEBUG") # uncomment for release builds

# include OpenCV for image-to-tensor
add_compile_definitions("USING_OPENCV")
add_compile_definitions("EIGEN_USE_THREADS")


# without this, nothing works
include_directories(lib/Eigen)

get_directory_property(DEFINITIONS COMPILE_DEFINITIONS)

if (DEFINITIONS MATCHES "USING_OPENCV")
    message("INCLUDING OPENCV")
    set(OpenCV_DIR lib/opencv)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message("NOT USING OPENCV")
endif ()

message(${CMAKE_BUILD_TYPE})

# ignore warning for virtual functions returning nothing
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
else ()
    set(MACOSX FALSE)
endif ()

add_subdirectory(orion)
add_subdirectory(examples)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})