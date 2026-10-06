//! Random nested pipelines (mostly sane parameters, some wild ones) on random samples.
#![no_main]

use augrs_fuzz::{Cfg, G, run_case};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut u = arbitrary::Unstructured::new(data);
    let cfg = Cfg {
        wild: 12,
        max_depth: 4,
        max_elems: 1 << 20,
    };
    let _ = run_case(&mut G::new(&mut u, cfg), 6);
});
