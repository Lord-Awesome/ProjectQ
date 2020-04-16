rm -rf build/
/home/jonaher/Python-3.8.1/python3.8 setup.py build
cp kernel1.so build/lib.linux-x86_64-3.8/projectq/backends/_sim/kernel1.so
cp examples/shor.py build/lib.linux-x86_64-3.8/shor.py