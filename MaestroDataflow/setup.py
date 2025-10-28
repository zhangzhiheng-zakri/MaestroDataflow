#!/usr/bin/env python3
"""
MaestroDataflow - A Modern Data Processing Pipeline Framework
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

    def find_packages(where='.', exclude=()):
        # 简易实现 find_packages 功能
        packages = []
        root = convert_path(where)
        for dirpath, dirnames, filenames in os.walk(root):
            if '__init__.py' in filenames:
                package = dirpath.replace(root, '').lstrip(os.sep).replace(os.sep, '.')
                if package and not any(e in package for e in exclude):
                    packages.append(package)
        return packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MaestroDataflow - A Modern Data Processing Pipeline Framework"

# Read requirements
def read_requirements():
    requirements = [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'sqlalchemy>=1.4.0',
        'pymysql>=1.0.0',
        'psycopg2-binary>=2.8.0',
        'openpyxl>=3.0.0',
        'pyarrow>=5.0.0',
        'requests>=2.25.0',
        'openai>=1.0.0',
        'transformers>=4.20.0',
        'torch>=1.12.0',
        'accelerate>=0.20.0',
        'sentence-transformers>=2.2.0',
        'scipy>=1.9.0',
        'Pillow>=9.0.0',
        'scikit-learn>=1.1.0',
    ]
    return requirements

setup(
    name="maestro-dataflow",
    version="1.0.0",
    author="MaestroDataflow Team",
    author_email="maestro@dataflow.ai",
    description="A Modern Data Processing Pipeline Framework with AI Integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/maestro-dataflow/MaestroDataflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.900',
        ],
        'full': [
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'maestro=maestro.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)