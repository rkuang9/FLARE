set(SOURCES
        fl_types.hpp
        flare.hpp
        fl_assert.hpp
        main.cpp
        sequential.cpp
        )

set(SOURCE_DIRECTORY)

add_subdirectory(activations)
add_subdirectory(layers)
add_subdirectory(weights)
add_subdirectory(optimizers)
add_subdirectory(loss)


# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)