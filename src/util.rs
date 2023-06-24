//! Utilities.

use ndarray::{self, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::{lobpcg::TruncatedOrder, TruncatedSvd};

use crate::Float;

/// Computes the cosine similarity in `[-1,1]`.
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

// The default value of maxiter will take a long time to converge, so we set a small value.
// (cf. https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/fse/models/utils.py)
const SVD_MAX_ITER: usize = 7;

/// Computes the principal components of the input matrix.
///
/// # Arguments
///
/// - `vectors`: 2D-array of shape `(n, m)`
/// - `n_components`: Number of components
///
/// # Returns
///
/// - Singular values of shape `(k,)`
/// - Right singular vectors of shape `(k, m)`
///
/// where `k` is the smaller one of `n_components` and `Rank(vectors)`.
pub(crate) fn principal_components<S>(
    vectors: &ArrayBase<S, Ix2>,
    n_components: usize,
) -> (Array1<Float>, Array2<Float>)
where
    S: Data<Elem = Float>,
{
    let n_components = n_components.min(vectors.ncols());
    let svd = TruncatedSvd::new(vectors.to_owned(), TruncatedOrder::Largest)
        .maxiter(SVD_MAX_ITER)
        .decompose(n_components)
        .unwrap();
    let (_, s, vt) = svd.values_vectors();
    (s, vt)
}

/// Removes the principal components from the input vectors,
/// returning the 2D-array of shape `(n, m)`.
///
/// # Arguments
///
/// - `vectors`: Sentence vectors to remove components from, of shape `(n, m)`
/// - `components`: `k` principal components of shape `(k, m)`
/// - `weights`: Weights of shape `(k,)`
pub(crate) fn remove_principal_components<S>(
    vectors: &ArrayBase<S, Ix2>,
    components: &ArrayBase<S, Ix2>,
    weights: Option<&ArrayBase<S, Ix1>>,
) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    // weighted_components of shape (k, m)
    let weighted_components = if let Some(weights) = weights {
        let weights = weights.to_owned().insert_axis(Axis(1));
        components * &weights
    } else {
        components.to_owned()
    };
    // projection of shape (m, m)
    let projection = weighted_components.t().dot(&weighted_components);
    vectors.to_owned() - &(vectors.dot(&projection))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_principal_components_k1() {
        let vectors = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let (s, vt) = principal_components(&vectors, 1);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 5]);
    }

    #[test]
    fn test_principal_components_k2() {
        let vectors = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let (s, vt) = principal_components(&vectors, 2);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 5]);
    }

    #[test]
    fn test_principal_components_k10() {
        let vectors = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let (s, vt) = principal_components(&vectors, 10);
        // Rank(x) = 3.
        assert_eq!(s.shape(), &[3]);
        assert_eq!(vt.shape(), &[3, 5]);
    }

    #[test]
    fn test_principal_components_zeros() {
        let vectors = ndarray::arr2(&[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]);
        let (s, vt) = principal_components(&vectors, 1);
        assert_eq!(s.shape(), &[0]);
        assert_eq!(vt.shape(), &[0, 5]);
    }

    #[test]
    fn test_principal_components_ones() {
        let vectors = ndarray::arr2(&[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
        ]);
        let (s, vt) = principal_components(&vectors, 1);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 5]);
    }

    #[test]
    fn test_remove_principal_components_k1() {
        let vectors = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 2., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 1., 0., 2., 2.],
        ]);
        let components = ndarray::arr2(&[
            [1., 1., 1., 0., 0.],
            [1., 2., 3., 4., 5.],
            [0., 1., 0., 3., 3.],
        ]);
        let weights = ndarray::arr1(&[1., 2., 4.]);
        let result = remove_principal_components(&vectors, &components, Some(&weights));
        assert_eq!(result.shape(), &[7, 5]);
    }
}
