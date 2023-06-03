//! Utilities.
use std::io::BufRead;

use anyhow::{anyhow, Result};
use ndarray::{self, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::{lobpcg::TruncatedOrder, TruncatedSvd};

use crate::Float;

/// Parses pairs of a word and its weight from a text file,
/// where each line has a word and its weight sparated by the ASCII whitespace.
///
/// ```text
/// <word> <weight>
/// ```
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sif_embedding::util;
///
/// let word_weight_text = "las 10\nvegas 20\n";
/// let word_weights = util::word_weights_from_text(word_weight_text.as_bytes())?;
///
/// assert_eq!(
///     word_weights,
///     vec![("las".to_string(), 10.), ("vegas".to_string(), 20.)]
/// );
/// # Ok(())
/// # }
/// ```
pub fn word_weights_from_text<R: BufRead>(rdr: R) -> Result<Vec<(String, Float)>> {
    let mut word_weights = vec![];
    for (i, line) in rdr.lines().enumerate() {
        let line = line?;
        let cols: Vec<_> = line.split_ascii_whitespace().collect();
        if cols.len() != 2 {
            return Err(anyhow!(
                "Line {i}: a line should be <word><SPACE><weight>, but got {line}."
            ));
        }
        word_weights.push((cols[0].to_string(), cols[1].parse()?));
    }
    Ok(word_weights)
}

/// Computes the cosine similarity.
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
fn right_singular_vectors<S>(x: &ArrayBase<S, Ix2>, k: usize) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    let svd = TruncatedSvd::new(x.to_owned(), TruncatedOrder::Largest)
        // NOTE(kampersanda): The default value of maxiter will take a long time to converge.
        // So, we set a small value, following https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/fse/models/utils.py.
        .maxiter(7)
        .decompose(k)
        .unwrap();
    let (_, _, vt) = svd.values_vectors();
    vt
}

/// Computes the principal components of the input data `x`,
/// returning a 2D-array of shape `(m, m)`.
///
/// This is direction c_0.
///
/// # Arguments
///
/// - `x`: 2D-array of shape `(n, m)`
/// - `k`: Number of components
pub(crate) fn principal_component<S>(x: &ArrayBase<S, Ix2>, k: usize) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    assert_ne!(k, 0);

    // NOTE(kampersanda): The description why the principal components are the right singular vectors can be found in
    // https://towardsdatascience.com/singular-value-decomposition-and-its-applications-in-principal-component-analysis-5b7a5f08d0bd

    // u of shape (k, m)
    let u = right_singular_vectors(x, k);
    // NOTE(kampersanda): Algorithm 1 says uu^T for a column vector u, not a row vector.
    // So, u^Tu is correct here.
    u.t().dot(&u)
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
        let y = principal_component(&x, 1);
        assert_eq!(y.shape(), &[5, 5]);
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
        let y = principal_component(&x, 2);
        assert_eq!(y.shape(), &[5, 5]);
    }
}
