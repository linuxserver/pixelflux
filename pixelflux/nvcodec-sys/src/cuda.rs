/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(deref_nullptr)]
#![allow(clippy::all)]

// CUDA driver bindings (the subset NVENC needs). Regenerated in place by build.rs under `regen`.
include!("bindgen/cuda.rs");
