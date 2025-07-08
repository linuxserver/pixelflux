import setuptools
from setuptools import setup, Extension

screen_capture_module = Extension(
    'pixelflux.screen_capture_module', 
    sources=[
        'pixelflux/screen_capture_module.cpp',
        'pixelflux/include/xxhash.c'
    ],
    include_dirs=['pixelflux/include'],
    libraries=['X11', 'Xext', 'Xfixes', 'jpeg', 'x264', 'yuv', 'dl', 'va', 'va-drm'],
    language='c++',
    extra_compile_args=['-std=c++17', '-Wno-unused-function']
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pixelflux",
    version="1.2.1",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",
    packages=setuptools.find_packages(),
    ext_modules=[screen_capture_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
