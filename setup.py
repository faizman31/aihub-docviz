from setuptools import setup, find_packages

setup(
    name='docviz',
    version='0.1.0',
    packages=find_packages(include=['docviz', 'docviz.*']),
    install_requires=[
        "transformers",
        "torch==2.4.0",
        "datasets",
        "pillow",
        "peft",
        "bitsandbytes",
        "numpy<2.0.0",
        "ipykernel",
        "ipywidgets",
        "jsonlines",
        "evaluate"
    ],
)