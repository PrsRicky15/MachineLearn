#[allow(dead_code)]
pub mod point3d {
    use std::ops::{Add, Sub, Mul};

    #[derive(Debug, Copy, Clone)]
    pub struct Coord3d<U> {
        x: U,
        y: U,
        z: U,
    }

    impl<U> Coord3d<U>
    where
        U: Add<Output = U> + Sub<Output = U> + Mul<Output = U> + Copy + Default + Into<f64>,
    {
        pub fn new(x: U, y: U, z: U) -> Self {
            Self { x, y, z }
        }

        pub fn shift_all(&self, a: U) -> Self {
            Self {
                x: self.x + a,
                y: self.y + a,
                z: self.z + a,
            }
        }

        pub fn shift_x(&self, a: U) -> Self {
            Self {
                x: self.x + a,
                y: self.y,
                z: self.z,
            }
        }

        pub fn shift_y(&self, a: U) -> Self {
            Self {
                x: self.x,
                y: self.y + a,
                z: self.z,
            }
        }

        pub fn shift_z(&self, a: U) -> Self {
            Self {
                x: self.x,
                y: self.y,
                z: self.z + a,
            }
        }

        pub fn scale(&self, a: U) -> Self {
            Self {
                x: self.x * a,
                y: self.y * a,
                z: self.z * a,
            }
        }

        pub fn add_coord(&self, coord3d: Coord3d<U>) -> Self {
            Self {
                x: self.x + coord3d.x,
                y: self.y + coord3d.y,
                z: self.z + coord3d.z,
            }
        }

        pub fn origin() -> Self {
            Self {
                x: U::default(),
                y: U::default(),
                z: U::default(),
            }
        }

        pub fn inner_product(&self) -> U {
            self.x * self.x + self.y * self.y + self.z * self.z
        }

        pub fn outter_product(&self, other: &Coord3d<U>) -> [[U; 3]; 3] {
            [
                [self.x * other.x, self.x * other.y, self.x * other.z],
                [self.y * other.x, self.y * other.y, self.y * other.z],
                [self.z * other.x, self.z * other.y, self.z * other.z],
            ]
        }

        pub fn distance(&self, coord3d: Coord3d<U>) -> f64 {
            let dx = self.x - coord3d.x;
            let dy = self.y - coord3d.y;
            let dz = self.z - coord3d.z;
            (dx * dx + dy * dy + dz * dz).into().sqrt()
        }
    }
}

pub mod blas{
    extern "C" {
        fn dgemm_(transa: *const u8, transb: *const u8, m: *const i32, n: *const i32, k: *const i32, alpha: *const f64,
            a: *const f64, lda: *const i32, b: *const f64, ldb: *const i32, beta: *const f64, c: *mut f64,
            ldc: *const i32);
    }

    pub fn dgemm(transa: char, transb: char, m: i32, n: i32, k: i32, alpha: f64, a: &[f64], lda: i32, b: &[f64],
        ldb: i32, beta: f64, c: &mut [f64], ldc: i32) {
        assert!(a.len() >= (lda * k) as usize, "Matrix A dimensions are incorrect.");
        assert!(b.len() >= (ldb * n) as usize, "Matrix B dimensions are incorrect.");
        assert!(c.len() >= (ldc * n) as usize, "Matrix C dimensions are incorrect.");
        unsafe {
            dgemm_(&(transa as u8), &(transb as u8), &m, &n, &k, &alpha, a.as_ptr(), &lda, b.as_ptr(), &ldb,
                &beta, c.as_mut_ptr(), &ldc);
        }
    }

}