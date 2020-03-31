from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='DIET-pytorch',
    version='0.1',
    description='Dual Intent Entity Transformer based on pytorch-lightning',
    author='Cheesama',
    install_requires=[],
    packages=find_packages(exclude=['docs','tests','tmp','data']),
    python_requires='>=3',
    zip_safe=False,
    include_package_data=True
)

