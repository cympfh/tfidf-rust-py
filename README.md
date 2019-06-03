# tfidf-rust-py

`tf-idf` library for Python written in Rust.

## Build `.so` library file

`rustc` 1.25.0+ required (`stable` is ok).

```bash
   rustc --version
rustc 1.32.0 (9fda7c223 2019-01-16)

   make
```

Either `target/release/libtfidf.so` or `target/release/libtfidf.dylib` will be produced
(If you are Mac user, you will find `dylib`.)

## import from Python

Copy the `so` (or `dylib`) file as `libtfidf.so`, and you can import it from Python.

```bash
   cp target/release/libtfidf.dylib ~/test/libtfidf.so  # on Mac
   # or:  cp target/release/libtfidf.so ~/test/libtfidf.so  # on Linux
   cd ~/test
```

```python
# ipython
In [1]: from libtfidf import tfidf

In [2]: corpus = [[0, 0, 1, 2], [0, 1, 3, 4], [1, 2]]l

# A corpus is a list of document.
# A document is a list of word.
# Word is natural integer (0,1,2,...).

In [3]: tfidf(corpus)
Out[3]:
(3,
 5,
 [(0, 0, 0.8109301924705505),
  (0, 2, 0.40546509623527527),
  (1, 0, 0.40546509623527527),
  (1, 3, 1.0986123085021973),
  (1, 4, 1.0986123085021973),
  (2, 2, 0.40546509623527527)])

# Output is a matrix, its shape is (document-size)x(vocaburary-size)

# This is 3x5 matrix
# The list in 3rd position is a list of (row, column, value)

# This matrix will be converted to scipy.sparse.csr_matrix easily

In [4]: from scipy.sparse import csr_matrix

In [5]: n, m, data = tfidf(corpus)

In [6]: rows, cols, values = zip(*data)

In [7]: csr_matrix((values, (rows, cols)), shape=(n, m))
Out[7]:
<3x5 sparse matrix of type '<class 'numpy.float64'>'
        with 6 stored elements in Compressed Sparse Row format>

In [8]: csr_matrix((values, (rows, cols)), shape=(n, m)).todense()
Out[8]:
matrix([[0.81093019, 0.        , 0.4054651 , 0.        , 0.        ],
        [0.4054651 , 0.        , 0.        , 1.09861231, 1.09861231],
        [0.        , 0.        , 0.4054651 , 0.        , 0.        ]])
```
