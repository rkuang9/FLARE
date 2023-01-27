set(SOURCES
        #binary_cross_entropy.cpp
        #mean_squared_error.cpp
        #categorical_cross_entropy.cpp
        #mean_absolute_error.cpp
        #kl_divergence.cpp
        )

set(SOURCE_DIRECTORY)

# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)