#!/bin/bash

MODEL_NAME=256_256_resfcn256_weight

if [ ! -e ./data/net-data/${MODEL_NAME}.data-00000-of-00001 ] ; then
  echo "Download the pretrained model manually (see README for the link)"
  echo "and put the downloaded file"
  echo "'256_256_resfcn256_weight.data-00000-of-00001'"
  echo "into ./data/net-data"
fi

# -----------------------------------------------------------
# Add virtualenv and dependencies for building TensorFlow pip
# -----------------------------------------------------------
if [ ! -d env ] ; then
  # apt install python3-venv
  # I use venv because it can determine the location of site-packages
  # correctly, it's needed to configure TF sources before starting build
  python3 -m venv ./env
  source env/bin/activate
  pip install pip six wheel mock
  pip install numpy==1.16.2
  pip install -U keras_applications==1.0.6 --no-deps
  pip install -U keras_preprocessing==1.0.5 --no-deps
  deactivate
fi

# --------------------------
# Download and install Bazel
# --------------------------
BAZEL_VER=0.19.2
BAZEL_INSTALLER=bazel-${BAZEL_VER}-installer-linux-x86_64.sh

if [ ! -e ./deps/$BAZEL_INSTALLER ]
then
  wget -O ./deps/$BAZEL_INSTALLER \
    https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VER/$BAZEL_INSTALLER
  chmod +x $BAZEL_INSTALLER
  ./$BAZEL_INSTALLER --user
fi


# ----------------------------------
# Clone and build TensorFlow for C++
# ----------------------------------
if [ ! -d ./deps/tensorflow ] ; then
  git clone \
    -b "v1.13.1" --single-branch --depth 1 \
    --recursive https://github.com/tensorflow/tensorflow.git \
    ./deps/tensorflow
fi

CPP11_ABI=0
ENABLE_TF_GPU=1 # CUDA. CuDNN and NCCL should be installed as well
CUDA_PATH=/usr/local/cuda
CUDA_COMPUTE=5.0
ENABLE_MKL=1 # Intels' MKL
DOWNLOAD_MKL=0 # apt-get install -t stretch-backports intel-mkl

if [ ! -e ./deps/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.s ] ; then
  source env/bin/activate

  export PATH="$PATH:$HOME/bin"
  cd ./deps/tensorflow

  export PYTHON_BIN_PATH=$(which python3)
  export PYTHON_LIB_PATH=$($PYTHON_BIN_PATH -c \
                           'import site; print(site.getsitepackages()[0])')
  export TF_NEED_ROCM=0
  export TF_NEED_AWS=0
  export TF_NEED_GCP=0
  export TF_NEED_HDFS=0
  export TF_NEED_OPENCL=0
  export TF_NEED_JEMALLOC=1
  export TF_ENABLE_XLA=0
  export TF_NEED_VERBS=0
  export TF_CUDA_CLANG=0
  export TF_NEED_MKL=$ENABLE_MKL
  export TF_DOWNLOAD_MKL=$ENABLE_MKL
  export TF_NEED_MPI=0
  export TF_NEED_S3=0
  export TF_NEED_KAFKA=0
  export TF_NEED_GDR=0
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_TENSORRT=0
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_NEED_GCP=0
  export TF_NEED_CUDA=$ENABLE_TF_GPU

  if [ $ENABLE_TF_GPU == 1 ] ; then
    export CUDA_TOOLKIT_PATH=$CUDA_PATH
    export CUDNN_INSTALL_PATH=$CUDA_PATH
    export NCCL_INSTALL_PATH=$CUDA_PATH

    get_def () {
      def=$1
      path=$2
      echo $(sed -n "s/^#define $def\s*\(.*\).*/\1/p" $path)
    }
    export TF_CUDA_VERSION=$($CUDA_TOOLKIT_PATH/bin/nvcc --version | \
                             sed -n 's/^.*release \(.*\),.*/\1/p')
    export TF_CUDA_COMPUTE_CAPABILITIES=$CUDA_COMPUTE
    export TF_CUDNN_VERSION=$(get_def CUDNN_MAJOR \
                              $CUDNN_INSTALL_PATH/include/cudnn.h)

    NCCL_MAJOR=$(get_def NCCL_MAJOR $CUDNN_INSTALL_PATH/include/nccl.h)
    NCCL_MINOR=$(get_def NCCL_MINOR $CUDNN_INSTALL_PATH/include/nccl.h)
    export TF_NCCL_VERSION=${NCCL_MAJOR}.${NCCL_MINOR}
  fi

  export GCC_HOST_COMPILER_PATH=$(which gcc)
  export CC_OPT_FLAGS="-march=native -D_GLIBCXX_USE_CXX11_ABI=${CPP11_ABI}"

  ./configure

  bazel build --config=opt //tensorflow:libtensorflow_cc.so

  # this takes quite a bit of time
  bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg

  deactivate

  cd ../..
fi


# -------------------------------------
# Copy TensorFlow libraries and headers
# -------------------------------------
if [ ! -d ./deps/tensorflow_dist ] ; then
  source ./env/bin/activate
  pip install ./deps/tensorflow/tensorflow_pkg/tensorflow*.whl

  TF_INFO=($(python -c "import tensorflow as tf; \
print(tf.__version__); \
print(tf.__cxx11_abi_flag__); \
print(tf.sysconfig.get_include()); \
print(tf.sysconfig.get_lib() + '/libtensorflow_framework.so');
"))

  TF_INCLUDE=${TF_INFO[2]}
  TF_LIB=${TF_INFO[3]}

  mkdir ./deps/tensorflow_dist
  cp -r $TF_INCLUDE ./deps/tensorflow_dist/

  mkdir -p ./deps/tensorflow_dist/include/tensorflow/cc/ops
  cp ./deps/tensorflow/bazel-genfiles/tensorflow/cc/ops/*.h \
        ./deps/tensorflow_dist/include/tensorflow/cc/ops/

  mkdir ./deps/tensorflow_dist/lib
  cp ./deps/tensorflow/bazel-bin/tensorflow/*.so ./deps/tensorflow_dist/lib

  find ./deps/tensorflow_dist -type d -print0 | xargs -0 chmod 0755
  find ./deps/tensorflow_dist -type f -print0 | xargs -0 chmod 0644

  deactivate
fi

# --------------------------------------
# Save the meta graph for loading in C++
# --------------------------------------
if [ ! -e ./data/net-data/${MODEL_NAME}.meta ] ; then
  source ./env/bin/activate
  ./save_graph.py
  deactivate
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
