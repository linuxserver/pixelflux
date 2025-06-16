import os
import shutil
import subprocess
import sys
from pathlib import Path
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

CMAKE_INSTALL_DIR = Path("/tmp/screen_capture_build_output")

def run_cmake_command(command_list, build_path):
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    command = " ".join(command_list)
    print(f"Running in {build_path}: {command}")
    try:
        subprocess.check_call(command, cwd=build_path, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"CMake command failed: {e}")
        sys.exit(1)

class CMakeBuildExt(build_ext):
    def run(self):
        cmake_args = ["-DCMAKE_INSTALL_PREFIX=" + str(CMAKE_INSTALL_DIR)]
        cmake_build_path = Path(__file__).parent / "src" / "build"
        cmake_configure_command = ["cmake", ".."] + cmake_args
        run_cmake_command(cmake_configure_command, str(cmake_build_path))
        cmake_build_command = [f"cmake --build . && cmake --install . --prefix {str(CMAKE_INSTALL_DIR)} --component screen_capture_runtime"]
        run_cmake_command(cmake_build_command, str(cmake_build_path))
    def get_outputs(self):
        so_path = CMAKE_INSTALL_DIR / "screen_capture" / "screen_capture_module.so"
        return [str(so_path)]

class CustomInstallLib(install_lib):
    def install(self):
        outputs = super().install()
        cmake_so_path = CMAKE_INSTALL_DIR / "screen_capture" / "screen_capture_module.so"
        lib_dir = Path(self.install_dir)
        install_so_path = lib_dir / "pixelflux" / "screen_capture_module.so"
        os.makedirs(str(install_so_path.parent), exist_ok=True)
        shutil.copy2(str(cmake_so_path), str(install_so_path))
        outputs.append(str(install_so_path))
        return outputs

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pixelflux",
    version="1.1.0",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_dir={"pixelflux": "pixelflux"},
    ext_modules=[
        Extension(
            "pixelflux.screen_capture_module",
            sources=[],
        )
    ],
    cmdclass={
        "build_ext": CMakeBuildExt,
        "install_lib": CustomInstallLib,
    },
    package_data={"pixelflux": ["*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
