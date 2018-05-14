#!/usr/bin/env bash

echo 'setup NMS and RoIAlign ...'

# Build NMS
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../


# Build RoIAlign
cd roialign/roi_align/src/cuda
echo 'Compiling crop_and_resize kernels by nvcc...'
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../../

