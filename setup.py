from skbuild import setup

setup(
    name="libcasm-mapping",
    version="2.0.1",
    packages=[
        "libcasm",
        "libcasm.mapping",
        "libcasm.mapping.info",
        "libcasm.mapping.mapsearch",
        "libcasm.mapping.methods",
    ],
    package_dir={"": "python"},
    cmake_install_dir="python/libcasm",
    include_package_data=False,
)
