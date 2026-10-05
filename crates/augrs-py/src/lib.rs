//! PyO3 bindings. The Python-facing API (transform classes, `Compose`, label
//! handling) lives in `python/augrs/__init__.py`; this module exposes a
//! `_Pipeline` built from a JSON spec, whose `apply`/`apply_batch` release the
//! GIL while augmenting.

use augrs_core::buffer::to_standard;
use augrs_core::{AugError, Buf, Input, Output, Pipeline};
use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn to_pyerr(e: AugError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A borrowed numpy array of one of the supported dtypes (keeps the borrow alive).
enum ArrIn<'py> {
    U8(PyReadonlyArray3<'py, u8>),
    U16(PyReadonlyArray3<'py, u16>),
    I32(PyReadonlyArray3<'py, i32>),
    F32(PyReadonlyArray3<'py, f32>),
}

/// A view into an `ArrIn`, safe to send to worker threads while the GIL is released.
#[derive(Clone, Copy)]
enum ViewIn<'a> {
    U8(ArrayView3<'a, u8>),
    U16(ArrayView3<'a, u16>),
    I32(ArrayView3<'a, i32>),
    F32(ArrayView3<'a, f32>),
}

impl<'py> ArrIn<'py> {
    fn extract(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<PyReadonlyArray3<'py, u8>>() {
            return Ok(ArrIn::U8(a));
        }
        if let Ok(a) = obj.extract::<PyReadonlyArray3<'py, f32>>() {
            return Ok(ArrIn::F32(a));
        }
        if let Ok(a) = obj.extract::<PyReadonlyArray3<'py, u16>>() {
            return Ok(ArrIn::U16(a));
        }
        if let Ok(a) = obj.extract::<PyReadonlyArray3<'py, i32>>() {
            return Ok(ArrIn::I32(a));
        }
        Err(PyTypeError::new_err(format!(
            "{what} must be a 3-D (H, W, C) numpy array of uint8, uint16, int32 or float32"
        )))
    }

    fn view(&self) -> ViewIn<'_> {
        match self {
            ArrIn::U8(a) => ViewIn::U8(a.as_array()),
            ArrIn::U16(a) => ViewIn::U16(a.as_array()),
            ArrIn::I32(a) => ViewIn::I32(a.as_array()),
            ArrIn::F32(a) => ViewIn::F32(a.as_array()),
        }
    }
}

impl ViewIn<'_> {
    fn to_buf(self) -> Buf {
        match self {
            ViewIn::U8(v) => Buf::U8(to_standard(v)),
            ViewIn::U16(v) => Buf::U16(to_standard(v)),
            ViewIn::I32(v) => Buf::I32(to_standard(v)),
            ViewIn::F32(v) => Buf::F32(to_standard(v)),
        }
    }
}

fn buf_to_py<'py>(py: Python<'py>, b: Buf) -> PyResult<Bound<'py, PyAny>> {
    Ok(match b {
        Buf::U8(a) => a.into_pyarray(py).into_any(),
        Buf::U16(a) => a.into_pyarray(py).into_any(),
        Buf::I32(a) => a.into_pyarray(py).into_any(),
        Buf::F32(a) => a.into_pyarray(py).into_any(),
    })
}

/// Read an `(N, k)` float64 array (k <= 4) into rows padded to 4 columns.
fn rows4(a: &PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<Vec<[f64; 4]>> {
    let shape = a.shape();
    let (n, k) = (shape[0], shape[1]);
    if k > 4 || (n > 0 && k < 2) {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (N, 2..4), got ({n}, {k})"
        )));
    }
    let v = a.as_array();
    let mut out = Vec::with_capacity(n);
    for row in v.rows() {
        let mut r = [0.0; 4];
        for (j, x) in row.iter().enumerate() {
            r[j] = *x;
        }
        out.push(r);
    }
    Ok(out)
}

fn rows_to_py<'py>(py: Python<'py>, rows: &[[f64; 4]], ncols: usize) -> Bound<'py, PyArray2<f64>> {
    let mut arr = ndarray::Array2::<f64>::zeros((rows.len(), ncols));
    for (i, r) in rows.iter().enumerate() {
        for j in 0..ncols {
            arr[[i, j]] = r[j];
        }
    }
    arr.into_pyarray(py)
}

fn output_tuple<'py>(py: Python<'py>, o: Output, kp_cols: usize) -> PyResult<Bound<'py, PyTuple>> {
    let image = buf_to_py(py, o.image)?;
    let masks: Vec<Bound<'py, PyAny>> = o.masks.into_iter().map(|m| buf_to_py(py, m)).collect::<PyResult<_>>()?;
    let bboxes = rows_to_py(py, &o.bboxes, 4).into_any();
    let kps = rows_to_py(py, &o.keypoints, kp_cols).into_any();
    let extra: Vec<Bound<'py, PyAny>> = o
        .extra_images
        .into_iter()
        .map(|m| buf_to_py(py, m))
        .collect::<PyResult<_>>()?;
    let applied = o.applied.map(|a| serde_json::to_string(&a).unwrap_or_default());
    let items: Vec<Bound<'py, PyAny>> = vec![
        image,
        masks.into_bound_py_any(py)?,
        bboxes,
        o.bbox_ids.into_bound_py_any(py)?,
        kps,
        o.keypoint_ids.into_bound_py_any(py)?,
        applied.into_bound_py_any(py)?,
        o.seed.into_bound_py_any(py)?,
        extra.into_bound_py_any(py)?,
    ];
    PyTuple::new(py, items)
}

fn pool(n: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut map = POOLS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(p) = map.get(&n) {
        return Ok(p.clone());
    }
    let p = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("augrs-{i}"))
        .build()
        .map_err(|e| PyValueError::new_err(format!("cannot build thread pool: {e}")))?;
    let p = Arc::new(p);
    map.insert(n, p.clone());
    Ok(p)
}

/// Low-level pipeline. Use `augrs.Compose` instead.
#[pyclass(name = "_Pipeline", module = "augrs._augrs", frozen)]
struct PyPipeline {
    inner: Pipeline,
    kp_cols: usize,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (spec_json, seed=None))]
    fn new(spec_json: &str, seed: Option<u64>) -> PyResult<Self> {
        let inner = Pipeline::from_json(spec_json, seed).map_err(to_pyerr)?;
        let kp_cols = inner
            .spec()
            .keypoint_params
            .as_ref()
            .map(|k| k.format.ncols())
            .unwrap_or(2);
        Ok(PyPipeline { inner, kp_cols })
    }

    /// Reset the random stream (`None` = reseed from entropy).
    #[pyo3(signature = (seed=None))]
    fn set_seed(&self, seed: Option<u64>) {
        self.inner.set_seed(seed);
    }

    fn next_seed(&self) -> u64 {
        self.inner.next_seed()
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Augment one sample. The GIL is released while the Rust code runs.
    /// Returns `(image, masks, bboxes, bbox_ids, keypoints, keypoint_ids, applied_json, seed, extra_images)`.
    #[pyo3(signature = (image, masks=Vec::new(), bboxes=None, keypoints=None, seed=None, record=false, extra_images=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        masks: Vec<Bound<'py, PyAny>>,
        bboxes: Option<PyReadonlyArray2<'py, f64>>,
        keypoints: Option<PyReadonlyArray2<'py, f64>>,
        seed: Option<u64>,
        record: bool,
        extra_images: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let img = ArrIn::extract(&image, "image")?;
        let ms: Vec<ArrIn> = masks
            .iter()
            .map(|m| ArrIn::extract(m, "mask"))
            .collect::<PyResult<_>>()?;
        let xs: Vec<ArrIn> = extra_images
            .iter()
            .map(|m| ArrIn::extract(m, "image"))
            .collect::<PyResult<_>>()?;
        let bb = match &bboxes {
            Some(b) => rows4(b, "bboxes")?,
            None => vec![],
        };
        let kp = match &keypoints {
            Some(k) => rows4(k, "keypoints")?,
            None => vec![],
        };
        let iv = img.view();
        let mvs: Vec<ViewIn> = ms.iter().map(|m| m.view()).collect();
        let xvs: Vec<ViewIn> = xs.iter().map(|m| m.view()).collect();
        let inner = &self.inner;
        let res = py.detach(move || {
            let inp = Input {
                image: iv.to_buf(),
                extra_images: xvs.iter().map(|v| v.to_buf()).collect(),
                masks: mvs.iter().map(|v| v.to_buf()).collect(),
                bboxes: bb,
                keypoints: kp,
            };
            let seed = seed.unwrap_or_else(|| inner.next_seed());
            inner.apply_with_seed(inp, seed, record)
        });
        output_tuple(py, res.map_err(to_pyerr)?, self.kp_cols)
    }

    /// Augment a batch in parallel (rayon) with the GIL released. Sample `i`
    /// uses a seed derived from `(seed, i)`, independent of `num_threads`.
    #[pyo3(signature = (images, masks=None, bboxes=None, keypoints=None, seed=None, num_threads=None, record=false, extra_images=None))]
    #[allow(clippy::too_many_arguments)]
    fn apply_batch<'py>(
        &self,
        py: Python<'py>,
        images: Vec<Bound<'py, PyAny>>,
        masks: Option<Vec<Vec<Bound<'py, PyAny>>>>,
        bboxes: Option<Vec<Option<PyReadonlyArray2<'py, f64>>>>,
        keypoints: Option<Vec<Option<PyReadonlyArray2<'py, f64>>>>,
        seed: Option<u64>,
        num_threads: Option<usize>,
        record: bool,
        extra_images: Option<Vec<Vec<Bound<'py, PyAny>>>>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let n = images.len();
        let check_len = |len: usize, what: &str| -> PyResult<()> {
            if len != n {
                return Err(PyValueError::new_err(format!(
                    "{what} has {len} entries but there are {n} images"
                )));
            }
            Ok(())
        };
        let imgs: Vec<ArrIn> = images
            .iter()
            .map(|m| ArrIn::extract(m, "image"))
            .collect::<PyResult<_>>()?;
        let nested = |v: &Option<Vec<Vec<Bound<'py, PyAny>>>>, what: &str| -> PyResult<Vec<Vec<ArrIn<'py>>>> {
            match v {
                Some(m) => {
                    check_len(m.len(), what)?;
                    m.iter()
                        .map(|v| v.iter().map(|x| ArrIn::extract(x, what)).collect::<PyResult<Vec<_>>>())
                        .collect()
                }
                None => Ok((0..n).map(|_| Vec::new()).collect()),
            }
        };
        let ms = nested(&masks, "masks")?;
        let xs = nested(&extra_images, "extra_images")?;
        let read_rows =
            |v: &Option<Vec<Option<PyReadonlyArray2<'py, f64>>>>, what: &str| -> PyResult<Vec<Vec<[f64; 4]>>> {
                match v {
                    Some(list) => {
                        check_len(list.len(), what)?;
                        list.iter()
                            .map(|o| match o {
                                Some(a) => rows4(a, what),
                                None => Ok(vec![]),
                            })
                            .collect()
                    }
                    None => Ok((0..n).map(|_| vec![]).collect()),
                }
            };
        let bbs = read_rows(&bboxes, "bboxes")?;
        let kps = read_rows(&keypoints, "keypoints")?;
        let ivs: Vec<ViewIn> = imgs.iter().map(|a| a.view()).collect();
        let mvs: Vec<Vec<ViewIn>> = ms.iter().map(|v| v.iter().map(|a| a.view()).collect()).collect();
        let xvs: Vec<Vec<ViewIn>> = xs.iter().map(|v| v.iter().map(|a| a.view()).collect()).collect();
        let inner = &self.inner;
        let pool = match num_threads {
            Some(t) if t > 0 => Some(pool(t)?),
            _ => None,
        };
        let results = py.detach(|| {
            let run = || {
                inner.apply_batch_with(n, seed, record, |i| {
                    Ok(Input {
                        image: ivs[i].to_buf(),
                        extra_images: xvs[i].iter().map(|v| v.to_buf()).collect(),
                        masks: mvs[i].iter().map(|v| v.to_buf()).collect(),
                        bboxes: bbs[i].clone(),
                        keypoints: kps[i].clone(),
                    })
                })
            };
            match &pool {
                Some(p) => p.install(run),
                None => run(),
            }
        });
        let mut out = Vec::with_capacity(n);
        for (i, r) in results.into_iter().enumerate() {
            match r {
                Ok(o) => out.push(output_tuple(py, o, self.kp_cols)?),
                Err(e) => return Err(PyValueError::new_err(format!("sample {i}: {e}"))),
            }
        }
        Ok(out)
    }
}

#[pymodule]
fn _augrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipeline>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
