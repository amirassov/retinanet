import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="retinanet",
    version="0.1",
    author="Miras Amir",
    author_email="amirassov@gmail.com",
    description="RetinaNet PyTorch implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amirassov/retinanet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    dependency_links=[
        'https://github.com/amirassov/youtrain#egg=package-0.1'
    ]
)
