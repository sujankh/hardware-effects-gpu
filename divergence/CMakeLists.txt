set(CMAKE_CXX_FLAGS_RELEASE "-O2")
cuda_add_executable(divergence init.cpp divergence.cu)
