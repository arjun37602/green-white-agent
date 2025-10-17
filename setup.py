from setuptools import setup, find_packages

setup(
    name="green-white-agent",
    version="0.1.0",
    description="Green agent for Terminal-Bench integration",
    packages=find_packages(),
    install_requires=[
        "terminal-bench>=1.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "datasets>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    python_requires=">=3.8",
    author="Green Agent Team",
    author_email="team@greenagent.com",
)
