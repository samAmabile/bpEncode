from setuptools import setup 
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
        Pybind11Extension(
            "bpEncode",
            ["bpe.cpp"],
        ),
]

setup(
        name="bpEncode",
        version="1.0.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
)

