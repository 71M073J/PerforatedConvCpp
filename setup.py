from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='perfconv',
    ext_modules=[
        CUDAExtension('perfconv',
                      sources=['perfconv.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from test_compile import test
test()
print("Test done, Succesful")