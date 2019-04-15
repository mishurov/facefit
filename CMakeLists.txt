cmake_minimum_required(VERSION 3.7)

project(FaceFitProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# It CXX11_ABI is not set, linking errors appear, Nuke 11.3v1
# seem compiled with that ABI. It should be set for TensorFlow configuration
# as well before starting bazel build
set(CPP11_ABI 0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CPP11_ABI}")

# The version of the installed Nuke i.e. 11.3v1
set(Nuke_MAJOR 11.3)
set(Nuke_MINOR 1)

# Nuke's installation path, may be in /opt/ or something if installed as root
set(Nuke_DIR ~/Nuke${Nuke_MAJOR}v${Nuke_MINOR})

# Dlib and TensorFlow are supposed
# to be downloaded by the dependencies.sh script
set(Dlib_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/dlib)
set(TensorFlow_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/tensorflow_dist)

# Dlib's dnn face detector doesn't fit into 2GB of my laptop's nvidia
# with set(DLIB_USE_CUDA OFF) it's very slow on CPU,
# so I use classic HOG detector

# Dlib on Debian also can't find cblas, I couldn't find a solution
# apt-get install -t stretch-backports liblapack3 \
#    liblapack-dev libblas-dev libatlas-base-dev
# It finds Intel MKL though, its benchmarks are the fastest
# apt-get install -t stretch-backports intel-mkl

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
#set(DLIB_USE_CUDA OFF)
add_subdirectory(${Dlib_DIR} dlib_build)
set_target_properties(dlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Dlib recommens to use SSE4 or AVX for an application
set(USE_AVX_INSTRUCTIONS ON)


include_directories(
    ${Nuke_DIR}/include
    ${TensorFlow_DIR}/include
)

link_directories(
    ${Nuke_DIR}
    ${TensorFlow_DIR}/lib
)

add_library(FaceFit SHARED
    src/facefit.cpp
    src/nuke2tf.cpp
    src/prnet.cpp
)


#add_executable(FaceFit src/facefit.cpp)

set_target_properties(FaceFit PROPERTIES PREFIX "")

target_link_libraries(FaceFit
    DDImage
    nuke-${Nuke_MAJOR}.${Nuke_MINOR}
    tensorflow_cc
    tensorflow_framework
    dlib::dlib
)


