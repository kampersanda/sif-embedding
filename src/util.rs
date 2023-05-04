use std::io::BufRead;

use anyhow::{anyhow, Result};
use ndarray::{self, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::SVD;

use crate::Float;

pub fn word_weights_from_text<R: BufRead>(rdr: R) -> Result<Vec<(String, Float)>> {
    let mut word_weights = vec![];
    for line in rdr.lines() {
        let line = line?;
        let cols: Vec<_> = line.split_ascii_whitespace().collect();
        if cols.len() != 2 {
            return Err(anyhow!(""));
        }
        word_weights.push((cols[0].to_string(), cols[1].parse()?));
    }
    Ok(word_weights)
}

pub fn cosine_similarity<S, T>(a: &ArrayBase<S, Ix1>, b: &ArrayBase<T, Ix1>) -> Option<Float>
where
    S: Data<Elem = Float>,
    T: Data<Elem = Float>,
{
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0. || norm_b == 0. {
        None
    } else {
        Some(dot_product / (norm_a * norm_b))
    }
}

/// Computes the right singular vectors of the input data `x`,
/// returning a 2D-array of shape `(k, m)`.
///
/// # Arguments
///
/// - `x`: 2D-array of shape `(n, m)`
/// - `k`: Number of components
pub fn right_singular_vectors<S>(x: &ArrayBase<S, Ix2>, k: usize) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    let (_, _, vt) = x.svd(false, true).unwrap();
    let vt = vt.unwrap();
    vt.slice(ndarray::s![..k, ..]).to_owned()
}

/// Computes the principal components of the input data `x`,
/// returning a 2D-array of shape `(k, k)`.
///
/// This is direction c_0.
///
/// # Arguments
///
/// - `x`: 2D-array of shape `(n, m)`
/// - `k`: Number of components
pub fn principal_components<S>(x: &ArrayBase<S, Ix2>, k: usize) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    let u = right_singular_vectors(x, k);
    u.dot(&u.t())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_right_singular_vectors_k1() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let y = right_singular_vectors(&x, 1);
        assert_eq!(y.shape(), &[1, 5]);
    }

    #[test]
    fn test_right_singular_vectors_k2() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let y = right_singular_vectors(&x, 2);
        assert_eq!(y.shape(), &[2, 5]);
    }

    #[test]
    fn test_principal_components_k1() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let y = principal_components(&x, 1);
        assert_eq!(y.shape(), &[1, 1]);
    }

    #[test]
    fn test_principal_components_k2() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let y = principal_components(&x, 2);
        assert_eq!(y.shape(), &[2, 2]);
    }
}
