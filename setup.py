import os
import platform
import subprocess
import setuptools
from setuptools import Extension, setup
from setuptools_rust import Binding, RustExtension, Strip

if "RUSTFLAGS" not in os.environ:
    machine = platform.machine()
    if machine == "x86_64":
        print("Enabling x86-64-v3 optimizations (AVX2/FMA)")
        os.environ["RUSTFLAGS"] = "-C target-cpu=x86-64-v3"


def _pkg_config(*args):
    """pkg-config output tokens, or [] when pkg-config / the .pc files are
    missing (caller then relies on the hardcoded -l flags + include_dirs)."""
    pkg_config = os.environ.get("PKG_CONFIG", "pkg-config")
    try:
        out = subprocess.check_output([pkg_config, *args], stderr=subprocess.DEVNULL)
        return out.decode(errors="surrogateescape").split()
    except (OSError, subprocess.CalledProcessError):
        return []


# Pull the toolchain include dirs (e.g. conda's $FLUX/include) from pkg-config so
# the C++ ext finds ffmpeg/x11/etc. headers reliably. Some setuptools versions
# drop CFLAGS' -I flags from the C++ compile, so we append pkg-config --cflags
# directly to extra_compile_args (and never remove the hardcoded -l fallback,
# keeping the existing build working when the .pc files are absent).
_pkg_cflags = _pkg_config(
    "--cflags", "x11", "xext", "xfixes", "libavcodec", "libavutil", "x264"
)

# Declarative C-API extension (full API, not Limited/abi3). xxhash.c is C but
# compiled as part of this C++ Extension; setuptools compiles each source per
# its own extension, so the .c stays C.
capture_ext = Extension(
    "pixelflux._capture",
    sources=["pixelflux/screen_capture_module.cpp", "pixelflux/include/xxhash.c"],
    include_dirs=["pixelflux/include"],
    libraries=["X11", "Xext", "Xfixes", "jpeg", "x264", "yuv", "dl", "avcodec", "avutil"],
    extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden", "-Wno-unused-function"]
    + _pkg_cflags,
    language="c++",
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
# The optional CUDA (NVRTC) color-conversion path loads `libnvrtc` at runtime; not auto-installed
# in install_requires because the correct version depends on the user's NVIDIA driver.
# `pixelflux[cuda]` opts in: cuda-toolkit's [nvrtc] extra pulls the right nvrtc wheel
# version-agnostically. Unpinned -> latest (CUDA 13); older drivers must pin (see README).
extras_require = {"cuda": ["cuda-toolkit[nvrtc]"]}

setup(
    name="pixelflux",
    install_requires=install_requires,
    extras_require=extras_require,
    version="1.6.4",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",
    packages=setuptools.find_packages(),

    ext_modules=[capture_ext],

    rust_extensions=[
        RustExtension(
            "pixelflux.pixelflux_wayland",
            "pixelflux_wayland/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            strip=Strip.All
        )
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    zip_safe=False,
)
