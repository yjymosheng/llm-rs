use core::f32;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}
// 几个切片可以进行优化, w可以提前被提取
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    assert!(y.shape().last() == x.shape().last() && x.shape().last() == w.shape().last());
    // assert!(y.shape() == x.shape());
    let len = y.size();
    assert!(len == x.size());
    let n_col = y.shape()[y.shape().len()-1];
    let n_row = y.shape()[y.shape().len()-2];
    let batch = y.size() / (n_col * n_row);
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let mut rec = vec![0.0f32; n_row];
    for b in 0..batch {
        let base = b * n_col * n_row;

        // 重置 rec 向量
        for j in 0..n_row {
            rec[j] = 0.0;
        }

        // 计算每一行的平方和
        for i in 0..(n_col * n_row) {
            let row_idx = i / n_col;
            rec[row_idx] += _x[i + base] * _x[i + base];
        }

        // 计算 RMS 并添加 epsilon
        for j in 0..n_row {
            rec[j] = (rec[j] / n_col as f32).sqrt() + epsilon;
        }

        // 归一化并应用权重
        for i in 0..(n_col * n_row) {
            let row_idx = i / n_col;
            let col_idx = i % n_col;
            _y[i + base] = _x[i + base] * w.data()[col_idx] / rec[row_idx];
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // let len = y.size();
    // assert!(len == x.size());

    assert!(y.size() == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    let y = unsafe { y.data_mut() };
    let x = x.data();
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(i, j)| (*i) *= (silu(*j)));
    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }
}
pub fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + (-1f32 * x).exp())
}

pub fn matmul_transb1(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 确保 A 和 B 能进行矩阵乘法
    assert!(a.shape().len() == b.shape().len());
    // 确保 A 和 C 能进行矩阵加法
    assert!(a.shape().len() == c.shape().len());

    let ndim = a.shape().len();
    assert!(ndim >= 2);
    let a_row = a.shape()[ndim - 2];
    let a_col = a.shape()[ndim - 1];

    let b_row = b.shape()[ndim - 2];
    let b_col = b.shape()[ndim - 1];

    let c_row = c.shape()[ndim - 2];
    let c_col = c.shape()[ndim - 1];

    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    assert!(a_col == b_col);
    assert!(c_col == b_row);
    assert!(a_row == c_row);

    for l in 0..c_row {
        for i in 0..c_col {
            let sum = (0..a_col)
                .map(|j| _a[l * a_col + j] * _b[i * b_col + j])
                .sum::<f32>();
            _c[l * c_col + i] = beta * _c[l * c_col + i] + alpha * sum;
        }
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
// 仅考虑 二维矩阵, 不会别的
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // println!("c[0][0]  {}", c.get(0, 0));
    // println!("c[0][1]  {}", c.get(0, 1));
    // println!("c[1][0]  {}", c.get(1, 0));
    // println!("c[1][1]  {}", c.get(1, 1));
    // c.print();

    let a_row = a.shape()[a.shape().len()-2];
    let a_col = a.shape()[a.shape().len()-1];
    let b_row = b.shape()[b.shape().len()-2];
    let b_col = b.shape()[b.shape().len()-1];
    let c_row = c.shape()[c.shape().len()-2];
    let c_col = c.shape()[c.shape().len()-1];
    assert!(a_col == b_col);
    assert!(a_row == c_row);
    assert!(b_row == c_col);
    // 计算batch
    let batch = c.size()/(c_row * c_col);
    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();
    let a_batch = a.size()/(a_row * a_row);
    let b_batch = b.size()/(b_row * b_row);
    // assert!(a_batch == b_batch);
    for j in _c.iter_mut() { *j *= beta; }
    for b in (0..batch) {
        let c_offset = b * c_row * c_col;
        let a_offset = b*a_row * a_col;
        let b_offset = b*b_row * b_col;
        for i in 0..c_row {
            for j in 0..c_col {
                // _c[i*c_col+j+c_offset]+=
            //     获取a，b来进行点击
                let ax = Tensor::new(
                    _a[(a_offset+i*a_col)..(a_offset+(i+1)*a_col)].to_vec(),
                    &vec![1,a_col]
                );
                let by = Tensor::new(
                    _b[(b_offset+j*b_col)..(b_offset+(j+1)*b_col)].to_vec(),
                    &vec![1,b_col]
                );
                let plus = dot(&ax,&by)*alpha;
                _c[i*c_col+j+c_offset]+=plus;
            }
        }
    }
    // c.print();
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
// 自行添加项 , 我怀疑我的矩阵乘法仅仅是巧合
// #[test]
// fn test_matmul_transb_2() {
//     let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.,5.,6.], &vec![2, 3]);
//     let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.,7.,8.], &vec![2, 4]);
//     let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.,7.,8.,9.,10.,11.,12.], &vec![3, 4]);
//     matmul_transb(&mut c, 1., &a, &b, 1.);
//     assert!(c.close_to(
//         &Tensor::<f32>::new(vec![31.,72.,113.,74.,179.,284.], &vec![2, 3]),
//         1e-3
//     ));
// }
