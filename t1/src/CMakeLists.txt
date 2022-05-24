ADD_EXECUTABLE(test test.cpp)
INCLUDE_DIRECTORIES( /usr/include/hello)
TARGET_LINK_LIBRARIES(test libhello.so)