import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='perfconv',
    ext_modules=[
        CUDAExtension('perfconv', sources=['perfconv.cpp'], extra_compile_args=["-O3"]) if torch.cuda.is_available()
        else CppExtension('perfconv', sources=['perfconv.cpp'], extra_compile_args=["-O3"])
    ],
    #extra_cflags=['-O3'],
    cmdclass={
        'build_ext': BuildExtension
    })
print("Done, testing...", flush=True)
from test_compile import test
test()
print("Test done, Succesful")