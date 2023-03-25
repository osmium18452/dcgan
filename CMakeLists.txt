cmake_minimum_required(VERSION 3.0)
project(dcgan)

set(CMAKE_CXX_STANDARD 14)
if (MSVC)
    set(CMAKE_PREFIX_PATH C:\\Users\\17110\\libtorch)
else()
    set(CMAKE_PREFIX_PATH /root/libtorch)
endif ()
message(${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)

add_executable(dcgan main.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET dcgan
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:dcgan>)
endif (MSVC)