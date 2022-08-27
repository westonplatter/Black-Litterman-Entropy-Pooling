import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

project_url = "https://github.com/westonplatter/Black-Litterman-Entropy-Pooling"

setuptools.setup(
    name="black_litterman_entropy_pooling",
    version="0.1.0",
    description="Black-Litterman Entropy Pooling Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Corey Hoffstein, Weston Platter',
    author_email='info@thinknewfound.com, westonplatter@gmail.com',
    license="MIT",
    url=project_url,
    python_requires=">=3.6",
    packages=["black_litterman_entropy_pooling"],
    install_requires=[
        "numpy",
        "pandas,
        "scipy",
    ],
    project_urls={
        "Issue Tracker": f"{project_url}/issues",
        "Source Code": f"{project_url}",
    },
)