set(SOURCES
        adam.cpp
        sgd.cpp
        rmsprop.cpp
        )

set(SOURCE_DIRECTORY)

# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)