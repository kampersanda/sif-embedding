//! Utilities.
use ndarray::{self, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::{lobpcg::TruncatedOrder, TruncatedSvd};

use crate::Float;

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
pub(crate) fn principal_components<S>(
    input: &ArrayBase<S, Ix2>,
    n_components: usize,
) -> (Array1<Float>, Array2<Float>)
where
    S: Data<Elem = Float>,
{
    // NOTE(kampersanda): The default value of maxiter will take a long time to converge.
    // So, we set a small value, following https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/fse/models/utils.py.
    let svd = TruncatedSvd::new(input.to_owned(), TruncatedOrder::Largest)
        .maxiter(7)
        .decompose(n_components)
        .unwrap();
    let (_, s, vt) = svd.values_vectors();
    (s, vt)
}

/// # Arguments
///
/// - `input`: 2D-array of shape `(n, m)`
/// - `components`: Principal components of shape `(k, m)`
/// - `weights`: Weights of shape `(k,)`
pub(crate) fn remove_principal_components<S>(
    input: &ArrayBase<S, Ix2>,
    components: &ArrayBase<S, Ix2>,
    weights: &ArrayBase<S, Ix1>,
) where
    S: Data<Elem = Float>,
{
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
    let (_, u) = principal_components(x, k);
    // NOTE(kampersanda): Algorithm 1 says uu^T for a column vector u, not a row vector.
    // So, u^Tu is correct here.
    u.t().dot(&u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_svd_k1() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let (s, vt) = principal_components(&x, 1);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 5]);
    }

    #[test]
    fn test_truncated_svd_k2() {
        let x = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let (s, vt) = principal_components(&x, 2);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 5]);
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
