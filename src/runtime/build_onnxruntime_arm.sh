#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates git curl unzip patch python3 python3-pip gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf make qemu-user
python3 -m pip install --no-cache-dir cmake==3.31.6 numpy==1.26.4
cd /work
if [ ! -d /opt/ort-linux-source/.git ]; then
  git clone --depth 1 --branch v1.20.1 --recursive https://github.com/microsoft/onnxruntime.git /opt/ort-linux-source
fi
test "$(git -C /opt/ort-linux-source rev-parse HEAD)" = 5c1b7ccbff7e5141c1da7a9d963d660e5741c319
# Microsoft's current deps.txt uses this mirror to avoid regenerated GitLab zips.
# Retain exactly the Eigen commit required by ORT 1.20.1; do not change versions.
if [ ! -d /opt/eigen-pinned/.git ]; then
  git init /opt/eigen-pinned
  git -C /opt/eigen-pinned remote add origin https://github.com/eigen-mirror/eigen.git
  git -C /opt/eigen-pinned fetch --depth 1 origin e7248b26a1ed53fa030c5c459f7ea095dfd276ac
  git -C /opt/eigen-pinned checkout --detach FETCH_HEAD
fi
test "$(git -C /opt/eigen-pinned rev-parse HEAD)" = e7248b26a1ed53fa030c5c459f7ea095dfd276ac
mkdir -p protoc
if [ ! -x protoc/bin/protoc ]; then
  curl -fL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip -o protoc.zip
  unzip -q -o protoc.zip -d protoc
fi
cat > arm-toolchain.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard")
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
cd /opt/ort-linux-source
python3 tools/ci_build/build.py --config MinSizeRel --update --build --parallel 2 --skip_tests \
  --build_shared_lib --disable_ml_ops --compile_no_warning_as_error --allow_running_as_root \
  --build_dir /opt/ort-build --path_to_protoc /work/protoc/bin/protoc \
  --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=/work/arm-toolchain.cmake \
    onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_ENABLE_LTO=OFF \
    onnxruntime_USE_PREINSTALLED_EIGEN=ON eigen_SOURCE_PATH=/opt/eigen-pinned
cp /opt/ort-build/MinSizeRel/libonnxruntime.so.1.20.1 /work/
cp include/onnxruntime/core/session/onnxruntime_c_api.h /work/
echo ONNX_RUNTIME_ARM_BUILD_COMPLETE
