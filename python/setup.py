from setuptools import setup


setup(
    name='tfidf',
    version='0.0.1',
    author='cympfh',
    author_email='cympfh@gmail.com',
    url='https://github.com/cympfh/tfidf-rust-py',
    install_requires=[
        'scipy==1.1.0',
    ],
    packages=['tfidf'],
    package_data={
        '': ['*.py', '*.so']
    },
)
