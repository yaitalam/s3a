import os
# cuda_toolkit_path = '/users/yaitalam/cuda_sys_arr/cuda-11.3'
# os.environ['CUDA_HOME'] = cuda_toolkit_path
# os.environ['PATH'] = f"{cuda_toolkit_path}/bin:" + os.environ['PATH']


import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='torch_systolic_array',
    version='1.0',
    author='yalama',
    author_email='yaitalam@uncc.edu',
    description="matmul to mimic systolic arrays",
    long_description="This c++/cuda extnetion is made to simulate how a systolic array would affect caulculations based on where a fault resides.",
    ext_modules=[
        CUDAExtension(
            name='torch_systolic_array',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    ## add gencode and arch for your gpu
                    '-gencode', 'arch=compute_70,code=sm_70',  # NVIDIA Titan V
                    '-gencode', 'arch=compute_75,code=sm_75',  # Titan RTX
                    '-gencode', 'arch=compute_80,code=sm_80',  # NVIDIA Tesla V100S NVIDIA A100
                    '-gencode', 'arch=compute_86,code=sm_86'   # NVIDIA A40/bunescu gpus
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)