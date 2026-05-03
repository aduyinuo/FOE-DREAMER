from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="foe_dreamer",
    version="0.1.0",
    description="Deployment-efficient world models with foe-aware auxiliary heads for network defense.",
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=requirements,
    python_requires=">=3.9",
    include_package_data=True,
)
