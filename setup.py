from setuptools import setup, find_packages

setup(
    name="employee-sentiment-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Employee Feedback Sentiment Analysis using RoBERTa Transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/employee-sentiment-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "streamlit>=1.12.0",
        "plotly>=5.10.0",
    ],
)
