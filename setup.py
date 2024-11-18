import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='perfconv',
    ext_modules=[
        CUDAExtension('perfconv', sources=['perfconv.cpp']) if torch.cuda.is_available()
        else CppExtension('perfconv', sources=['perfconv.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from test_compile import test
test()
print("Test done, Succesful")