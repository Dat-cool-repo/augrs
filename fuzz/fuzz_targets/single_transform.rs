//! One or two transforms with extreme parameters on degenerate images and targets.
#![no_main]

use augrs_fuzz::{Cfg, G, run_case};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut u = arbitrary::Unstructured::new(data);
    let cfg = Cfg {
        wild: 80,
        max_depth: 1,
        max_elems: 1 << 21,
    };
    let _ = run_case(&mut G::new(&mut u, cfg), 2);
});
