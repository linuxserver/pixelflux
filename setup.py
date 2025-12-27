import os
import subprocess
import sys
import platform
from pathlib import Path
import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools_rust import Binding, RustExtension, Strip

if "RUSTFLAGS" not in os.environ:
    machine = platform.machine()
    if machine == "x86_64":
        print("Enabling x86-64-v3 optimizations (AVX2/FMA)")
        os.environ["RUSTFLAGS"] = "-C target-cpu=x86-64-v3"

class BuildCtypesExt(build_ext):
    def run(self):
        super().run()
        self.build_custom_cpp()

    def build_custom_cpp(self):
        compiler = "g++"
        if hasattr(self, 'compiler') and self.compiler:
             if hasattr(self.compiler, 'compiler_cxx'):
                 compiler = self.compiler.compiler_cxx[0]

        lib_dir = Path(self.build_lib)
        output_path = lib_dir / "pixelflux" / "screen_capture_module.so"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sources = [
            'pixelflux/screen_capture_module.cpp',
            'pixelflux/include/xxhash.c'
        ]
        
        include_dirs = ['pixelflux/include']
        library_dirs = []
        
        if os.environ.get("CIBUILDWHEEL"):
            include_dirs.append('/usr/local/include')
            library_dirs.append('/usr/local/lib')

        libraries = ['X11', 'Xext', 'Xfixes', 'jpeg', 'x264', 'yuv', 'dl', 'avcodec', 'avutil']
        extra_compile_args = ['-std=c++17', '-Wno-unused-function', '-fPIC', '-O3', '-flto', '-shared']
            
        command = [compiler] + extra_compile_args + ['-o', str(output_path)]
        for inc in include_dirs: command.append(f'-I{inc}')
        for lib in library_dirs: command.append(f'-L{lib}')
        command.extend(sources)
        for lib in libraries: command.append(f'-l{lib}')
            
        print(f"Building C++ module: {' '.join(command)}")
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            print(f"C++ build failed with exit code {e.returncode}")
            sys.exit(1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
is_alpine = os.path.exists("/etc/alpine-release")
if not is_alpine:
    install_requires.append("nvidia-cuda-nvrtc")

setup(
    name="pixelflux",
    install_requires=install_requires,
    version="1.5.2",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",
    packages=setuptools.find_packages(),
    
    rust_extensions=[
        RustExtension(
            "pixelflux.pixelflux_wayland", 
            "pixelflux_wayland/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            strip=Strip.All
        )
    ],
    
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
    zip_safe=False,
)
