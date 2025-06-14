from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="synthiquery",
    version="0.1.0",
    author="Nithin",
    author_email="nithinganiga959@gmail.com",
    description="A RAG chatbot for PDF documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nithin-ganiga/rag-chatbot-ml",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "synthiquery=run:main",
        ],
    },
) 