#!/bin/bash

if [ ! -d ./deps ] ; then
  mkdir -p deps
fi

# ---------------------------------------
# Clone and build TensorFlow Lite for C++
# ---------------------------------------
if [ ! -d ./deps/tensorflow ] ; then
  git clone \
    -b "v1.13.1" --single-branch --depth 1 \
    --recursive https://github.com/tensorflow/tensorflow.git \
    ./deps/tensorflow
fi

if [ ! -e ./deps/tf_lite/lib/libtensorflow-lite.so ] ; then
  TFROOT=$(pwd)/deps/tensorflow
  cd $TFROOT/tensorflow/lite/tools/make
  ./download_dependencies.sh
  # remove NNAPI
  sed -i \
    's/tensorflow\/lite\/nnapi_delegate_disabled\.cc/tensorflow\/lite\/nnapi_delegate.cc/g' \
    Makefile
  # add -fPIC for C sources
  sed -i '/CCFLAGS := ${CXXFLAGS}/a CCFLAGS += -fPIC' Makefile

  cd $TFROOT
  make -j 3 -f tensorflow/lite/tools/make/Makefile \
    TARGET=linux TARGET_ARCH=x86_64 \
    EXTRA_CXXFLAGS="-march=native -DTF_LITE_USE_CBLAS \
-DTF_LITE_DISABLE_X86_NEON \
-D_GLIBCXX_USE_CXX11_ABI=0"

  # copy libraries and headers
  mkdir -p $TFROOT/../tf_lite/include/tensorflow

  cd $TFROOT/tensorflow
  find lite -name \*.h -exec cp --parents {} \
    $TFROOT/../tf_lite/include/tensorflow \;
  cp -r $TFROOT/tensorflow/lite/tools/make/downloads/flatbuffers/include/flatbuffers \
    $TFROOT/../tf_lite/include/
  cp -r $TFROOT/tensorflow/lite/tools/make/gen/linux_x86_64/lib \
    $TFROOT/../tf_lite/

  cd $TFROOT/../..
fi


# ----------
# Clone Dlib
# ----------
if [ ! -d ./deps/dlib ] ; then
  git clone \
    -b "v19.17" --single-branch --depth 1 \
    --recursive https://github.com/davisking/dlib.git \
    ./deps/dlib
fi
