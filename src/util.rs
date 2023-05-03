use ndarray::{self, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::SVD;

use crate::Float;

/// Computes the right singular vectors of the input data.
///
/// # Args
///
/// - `x`: 2D-array of shape (n_samples, n_features)
/// - `k`: n_components
///
/// # Returns
///
/// 2D-array of shape (n_components, n_features)
pub fn right_singular_vectors<S>(x: &ArrayBase<S, Ix2>, k: usize) -> Array2<Float>
where
    S: Data<Elem = Float>,
{
    let (_, _, vt) = x.svd(false, true).unwrap();
    let vt = vt.unwrap();
    vt.slice(ndarray::s![..k, ..]).to_owned()
}

/// Direction c_0
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
    fn test_right_singular_vectors() {
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
}
