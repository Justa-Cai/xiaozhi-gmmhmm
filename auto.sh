set -e
rm -rf build
mkdir -p build
cd build
cmake ../sdk
make -j `nproc`
cd ..

./build/demo/kws_demo
