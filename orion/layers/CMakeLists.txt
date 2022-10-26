set(SOURCES
        layer.hpp
        dense.ipp
        embedding.cpp
        globalaveragepooling1d.cpp
        conv2d.ipp
        maxpooling2d.cpp
        flatten.ipp
        )

set(SOURCE_DIRECTORY)

# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)
