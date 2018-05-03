TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all:
	g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2