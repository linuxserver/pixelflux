# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from setuptools import setup
from setuptools_rust import Binding, RustExtension, Strip

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pixelflux",
    version="1.6.4",
    author="Linuxserver.io",
    author_email="pypi@linuxserver.io",
    description="A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MPL-2.0",
    url="https://github.com/linuxserver/pixelflux",

    # Single self-contained Rust extension: the top-level `pixelflux` module does X11 (XShm)
    # and Wayland capture plus all encoding/conversion. No C/C++ sources, no Python package
    # layer -- `import pixelflux` resolves directly to pixelflux.cpython-*.so.
    packages=[],
    rust_extensions=[
        RustExtension(
            "pixelflux",
            "pixelflux/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            strip=Strip.All
        )
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    zip_safe=False,
)
