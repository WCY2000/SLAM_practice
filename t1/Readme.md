To install the static library & shared library:
cd lib
cmake -DCMAKE_INSTALL_PREFIX=usr/
make
make install


To compile the test:
cd build
cmake ..
make

