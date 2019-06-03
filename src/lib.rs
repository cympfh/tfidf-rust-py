#[macro_use]
extern crate cpython;

use cpython::{PyResult, Python};

py_module_initializer!(libtfidf, initlibtfidf, PyInit_libtfidf, |py, m| {
    m.add(py, "__doc__", "tf-idf")?;
    m.add(py, "tfidf", py_fn!(py, tfidf_py(docs: Vec<Vec<usize>>)))?;
    m.add(py, "tf", py_fn!(py, tf_py(docs: Vec<Vec<usize>>)))?;
    Ok(())
});

type SparseMatrix = (usize, usize, Vec<(usize, usize, f32)>);

fn tfidf(docs: &Vec<Vec<usize>>) -> SparseMatrix {
    use std::collections::{HashSet, HashMap};

    let docsize = docs.len();
    let vocsize = docs.iter().map(|d| d.iter().max().unwrap_or(&0)).max().unwrap_or(&0) + 1;
    let mut idf = vec![0.0f32; vocsize];

    // document freq
    for d in docs.iter() {
        let ws: HashSet<&usize> = d.iter().collect();
        for &w in ws.iter() {
            idf[*w] += 1.0;
        }
    }

    // inverse df
    for i in 0..vocsize {
        if idf[i] > 0.0 {
            idf[i] = (docsize as f32 / idf[i]).ln()
        }
    }

    let mut tfidf = vec![];

    // * tf
    for i in 0..docsize {
        let mut ws = HashMap::new();
        for &w in docs[i].iter() {
            let count = ws.entry(w).or_insert(0);
            *count += 1;
        }
        for (&w, &count) in ws.iter() {
            if idf[w] == 0.0 || count == 0 { continue }
            tfidf.push((i, w, idf[w] * count as f32));
        }
    }

    (docsize, vocsize, tfidf)
}

fn tfidf_py(_: Python, docs: Vec<Vec<usize>>) -> PyResult<SparseMatrix> {
    let out = tfidf(&docs);
    Ok(out)
}


fn tf(docs: &Vec<Vec<usize>>) -> SparseMatrix {
    use std::collections::HashMap;

    let docsize = docs.len();
    let vocsize = docs.iter().map(|d| d.iter().max().unwrap_or(&0)).max().unwrap_or(&0) + 1;

    let mut tf = vec![];

    // * tf
    for i in 0..docsize {
        let mut ws = HashMap::new();
        for &w in docs[i].iter() {
            let count = ws.entry(w).or_insert(0);
            *count += 1;
        }
        for (&w, &count) in ws.iter() {
            if count == 0 { continue }
            tf.push((i, w, count as f32));
        }
    }

    (docsize, vocsize, tf)
}

fn tf_py(_: Python, docs: Vec<Vec<usize>>) -> PyResult<SparseMatrix> {
    let out = tf(&docs);
    Ok(out)
}
