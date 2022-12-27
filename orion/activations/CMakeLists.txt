set(SOURCES
        sigmoid.hpp
        tanh.hpp
        relu.hpp
        softmax.hpp
        linear.hpp
        swish.hpp
        softplus.hpp
        softsign.hpp
        selu.hpp
        elu.hpp
        )

set(SOURCE_DIRECTORY)

# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)