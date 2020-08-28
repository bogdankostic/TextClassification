import setuptools

def parse_requirements(filename):
    """
    Parse a requirements pip file returning the list of required packages.
    It exclude commented lines.
    Args:
        filename: pip requirements requirements
    Returns:
        list of required package with versions constraints
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()
    parsed_requirements = [line.strip()
                           for line in parsed_requirements
                           if not ((line.strip()[0] == "#"))]
    return parsed_requirements


with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="text_classification",
    version="0.0.1",
    author="Bogdan Kostić",
    description="A text classification library.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    python_requires=">=3.6"
)
