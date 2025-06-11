"""
Setup configuration for AI Lead Enhancer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-lead-enhancer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered lead scoring and enrichment tool for SaaSQuatchLeads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-lead-enhancer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lead-enhancer-api=src.api.main:run_server",
            "lead-enhancer-ui=src.ui.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_lead_enhancer": [
            "data/*.csv",
            "models/*.pkl",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-lead-enhancer/issues",
        "Source": "https://github.com/yourusername/ai-lead-enhancer",
        "Documentation": "https://github.com/yourusername/ai-lead-enhancer/wiki",
    },
    keywords=[
        "lead-generation",
        "ai",
        "machine-learning",
        "sales-automation",
        "lead-scoring",
        "data-enrichment",
        "nlp",
        "saas",
    ],
)