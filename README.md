# tfidf-rust-py

`tf-idf` library for Python written in Rust.

## Build `.so` file

For building, `rustc` 1.25.0+ required (`stable` is ok).

```bash
   rustc --version
rustc 1.32.0 (9fda7c223 2019-01-16)

   make build
```

Either `target/release/libtfidf.so` or `target/release/libtfidf.dylib` will be produced.
If you are Mac user, you will find `dylib`.

## Install as Python library

Just type

```bash
   make install
```

This command copies the `so` file and an interface `__init__.py` to your path directory.

## How to use

```python
In [1]: import tfidf

In [2]: docs = [['a', 'a', 'b', 'c'], ['a', 'b', 'd', 'e'], ['b', 'c']]

# define your documents
# documents = list[document]
# document = list[word]
# word is Any (but hashable) value (int, str...)

In [3]: corpus = tfidf.Corpus(docs)

# make corpus
# (This process converts words in documents to int)

In [4]: tfidf.tfidf(corpus)
Out[4]:
<3x5 sparse matrix of type '<class 'numpy.float64'>'
        with 6 stored elements in Compressed Sparse Row format>

# tfidf.tfidf computes tf-idf (csr-)matrix

In [5]: Out[4].todense()
Out[5]:
matrix([[0.81093019, 0.        , 0.4054651 , 0.        , 0.        ],
        [0.4054651 , 0.        , 0.        , 1.09861231, 1.09861231],
        [0.        , 0.        , 0.4054651 , 0.        , 0.        ]])

# csr-matrix can be converted to numpy.matrix

# you can see the max value is (0, 0) in 0-th row
# This means that 0-th word in the 0-th document has highest tf-idf.
# Let's get 0-th word

In [6]: corpus.voc.get(0)
Out[6]: 'a'
```
