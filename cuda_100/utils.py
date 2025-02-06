import os
import torch
import glob

import setuptools
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


def get_src_files(
    src_path: str,
    file_extension_glob: str = "*.cpp",
) -> list:
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"invalid path: {src_path}")
    return list(
        glob.glob(
            os.path.join(src_path, "**", file_extension_glob),
            recursive=True,
        )
    )


def get_cxx_flags(
    debug_mode: bool = False,
) -> list[str]:
    return [
        "-std=c++17",
        "-O3" if not debug_mode else "-O0",
        "-g" if debug_mode else "",
    ]


def get_nvcc_flags(
    debug_mode: bool = False,
) -> list[str]:
    return [
        "-std=c++17",
        "-O3" if not debug_mode else "-O0",
        "-g" if debug_mode else "",
    ]


def get_extensions(
    src_path: str,
    extension_name: str,
    debug_mode: bool = False,
    use_cuda: bool = True,
) -> list[setuptools.Extension]:
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": get_cxx_flags(debug_mode=debug_mode),
        "nvcc": get_nvcc_flags(debug_mode=debug_mode),
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    sources = get_src_files(src_path, file_extension_glob="*.cpp")
    cuda_sources = get_src_files(src_path, file_extension_glob="*.cu")

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{extension_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules
