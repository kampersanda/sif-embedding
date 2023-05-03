use ndarray::{arr1, arr2};

#[test]
fn test_sum_2d() {
    let mut x = arr2(&[[0, 1], [2, 3], [4, 5]]);
    let t = arr1(&[10, 20]);
    for mut r in x.rows_mut() {
        r += &t;
    }
    assert_eq!(x, arr2(&[[10, 21], [12, 23], [14, 25]]));
}

#[test]
fn test_sum_2d_cast() {
    let mut x = arr2(&[[0, 1], [2, 3], [4, 5]]);
    let t = arr1(&[10]);
    for mut r in x.rows_mut() {
        r += &t;
    }
    assert_eq!(x, arr2(&[[10, 11], [12, 13], [14, 15]]));
}

#[test]
#[should_panic]
fn test_sum_2d_panic_1() {
    let mut x = arr2(&[[0, 1], [2, 3], [4, 5]]);
    let t = arr1(&[10, 20, 30]);
    for mut r in x.rows_mut() {
        r += &t;
    }
}

#[test]
#[should_panic]
fn test_sum_2d_panic_2() {
    let mut x = arr2(&[[0, 1, 2], [3, 4, 5], [5, 6, 7]]);
    let t = arr1(&[10, 20]);
    for mut r in x.rows_mut() {
        r += &t;
    }
}
