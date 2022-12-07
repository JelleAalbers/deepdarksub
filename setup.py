import setuptools

with open('requirements.txt') as f:
    requires= [x.strip() for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='deepdarksub',
    version='0.0.1',
    description='Dark matter substructure inference with ML',
    url='https://github.com/JelleAalbers/deepdarksub',
    python_requires=">=3.8",
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=['pytest'],
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    zip_safe=False)
