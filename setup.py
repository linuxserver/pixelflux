import setuptools
from setuptools import Extension, setup
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Set up the extension with proper build flags
extensions = [
    Extension(
        name='pixelflux.screen_capture',
        sources=[
            'pixelflux/screen_capture.pyx',
            'pixelflux/include/xxhash.c'
        ],
        include_dirs=['pixelflux/include'],
        libraries=['X11', 'Xext', 'Xfixes', 'jpeg', 'x264', 'va', 'va-drm'],
        extra_compile_args=['-std=c++17', '-fPIC', '-O3', '-Wno-unused-function'],
        language='c++',
    )
]

setup(
    name="pixelflux",
    version="1.2.8",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
