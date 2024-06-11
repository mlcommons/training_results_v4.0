cd /workspace
mkdir 550.27 && pushd 550.27 && wget http://linuxqa.nvidia.com/builds/release/display/x86_64/550.27/NVIDIA-Linux-x86_64-550.27.run && bash NVIDIA-Linux-x86_64-550.27.run --extract-only && popd
mkdir libcuda-550.27 && cp 550.27/NVIDIA-Linux-x86_64-550.27/libcuda.so.550.27 libcuda-550.27/
ln -s libcuda.so.550.27 libcuda-550.27/libcuda.so.1
ln -s libcuda.so.1 libcuda-550.27/libcuda.so

