import os
import shutil
import subprocess
import sys
from pathlib import Path
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class BuildCtypesExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_cxx[0]

        lib_dir = Path(self.build_lib)
        
        output_path = lib_dir / "pixelflux" / "screen_capture_module.so"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sources = [
            'pixelflux/screen_capture_module.cpp',
            'pixelflux/include/xxhash.c'
        ]
        include_dirs = ['pixelflux/include']
        libraries = ['X11', 'Xext', 'Xfixes', 'jpeg', 'x264', 'yuv', 'dl', 'va', 'va-drm']
        extra_compile_args = ['-std=c++17', '-Wno-unused-function', '-fPIC', '-g']

        command = [compiler]
        command.extend(extra_compile_args)
        command.append('-shared')
        command.append('-o')
        command.append(str(output_path))
        
        for include_dir in include_dirs:
            command.append(f'-I{include_dir}')
            
        command.extend(sources)

        for lib in libraries:
            command.append(f'-l{lib}')

        print("Running build command:")
        print(" ".join(command))
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            print(f"Build failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(1)

        print(f"Successfully built {output_path}")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pixelflux",
    version="1.2.5",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",
    packages=setuptools.find_packages(),
    
    ext_modules=[Extension("pixelflux.screen_capture_module", sources=[])],
    
    cmdclass={
       "build_ext": BuildCtypesExt,
    },

    package_data={
       "pixelflux": ["screen_capture_module.so"],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
