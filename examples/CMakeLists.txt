set(SOURCES
        xor_classifier.hpp
        diabetes_logistic_regression.hpp
        cnn_test.hpp
        )

set(SOURCE_DIRECTORY)

# add directory to files
foreach (file ${SOURCES})
    set(SOURCE_DIRECTORY ${SOURCE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach ()

set(SRC_FILES ${SRC_FILES} ${SOURCE_DIRECTORY} PARENT_SCOPE)