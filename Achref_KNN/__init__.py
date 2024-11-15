from setuptools import setup, find_packages

setup(
    name='Achref_KNN',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy'],
    description='Ma propre implémentation du classificateur KNN',
    author='Ton Nom',
    author_email='ton.email@example.com',
    url='https://github.com/Achrefdjaber/Achref_KNN',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)