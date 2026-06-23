/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Regenerates the committed NVENC + CUDA bindings from the bundled NVIDIA SDK headers in
//! `headers/`. Runs ONLY under the `regen` feature (which needs libclang); a normal build is a
//! no-op and compiles the checked-in `src/bindgen/*.rs`, so end-user builds need no libclang.
//!
//! bindgen does not emit function-like macros, so the NVENC struct-version constants
//! (`NV_ENC_*_VER`) and the `static const GUID` codec/preset GUIDs are extracted from the header
//! with regexes and appended -- mirroring how the upstream SDK defines them.

#[cfg(feature = "regen")]
fn main() -> std::io::Result<()> {
    use std::io::Write;
    let out = std::path::PathBuf::from("src/bindgen");
    std::fs::create_dir_all(&out)?;

    // ---- NVENC ----
    let nvenc_header = "headers/nvEncodeAPI.h";
    let nvenc_out = out.join("nvenc.rs");
    bindgen::builder()
        .header(nvenc_header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_type("NV.*")
        .allowlist_function("Nv.*")
        .allowlist_var("NVENC.*")
        .allowlist_var("NV_MAX.*")
        .size_t_is_usize(true)
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        // Newtype (not rustified) for enums whose values arrive from the driver -- as return
        // codes or in structs it fills that we then read/copy -- and for zero-less enums the
        // generated zero-filling Default impls materialize: an out-of-range discriminant in a
        // rustified enum is UB, and a newer driver may legally send codes this header predates.
        .newtype_enum("_NVENCSTATUS")
        .newtype_enum("_NV_ENC_PIC_TYPE")
        .newtype_enum("_NV_ENC_PIC_STRUCT")
        .newtype_enum("_NV_ENC_PARAMS_FRAME_FIELD_MODE")
        .newtype_enum("_NV_ENC_PARAMS_RC_MODE")
        .newtype_enum("_NV_ENC_MULTI_PASS")
        .newtype_enum("_NV_ENC_MV_PRECISION")
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .generate()
        .expect("Unable to generate NVENC bindings")
        .write_to_file(&nvenc_out)
        .expect("Unable to write nvenc.rs");

    let hdr = std::fs::read_to_string(nvenc_header)?;
    let mut extra = String::from(
        "\nconst fn nv_struct_version(ver: u32) -> u32 {\n    NVENCAPI_VERSION | ((ver) << 16) | (0x7 << 28)\n}\n",
    );
    // The high-bit OR term is written `( 1u<<31 )` in the SDK headers; capture just the shift
    // count so it can be re-emitted as a valid Rust literal (the C `1u` suffix is not Rust).
    let ver_re = regex::Regex::new(
        r"#define\s+([A-Z_]+)\s+\(?NVENCAPI_STRUCT_VERSION\((\d+)\)(?:\s*\|\s*\(\s*1u?\s*<<\s*(\d+)\s*\))?\s*\)?",
    )
    .unwrap();
    for c in ver_re.captures_iter(&hdr) {
        let (name, ver) = (&c[1], &c[2]);
        match c.get(3) {
            Some(shift) => extra.push_str(&format!(
                "pub const {}: u32 = nv_struct_version({}) | (1u32 << {});\n",
                name,
                ver,
                shift.as_str()
            )),
            None => extra.push_str(&format!(
                "pub const {}: u32 = nv_struct_version({});\n",
                name, ver
            )),
        }
    }
    let guid_re = regex::Regex::new(
        r"static\s+const\s+GUID\s+([A-Z_\d]+)\s*=\s*\r?\n\{\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*\{\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*,\s*(0[xX][0-9a-fA-F]+)\s*\}\s*\}\s*;",
    )
    .unwrap();
    for c in guid_re.captures_iter(&hdr) {
        extra.push_str(&format!(
            "pub const {}: GUID = GUID {{\n    Data1: {},\n    Data2: {},\n    Data3: {},\n    Data4: [{}, {}, {}, {}, {}, {}, {}, {}],\n}};\n",
            &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7], &c[8], &c[9], &c[10], &c[11], &c[12]
        ));
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(&nvenc_out)?
        .write_all(extra.as_bytes())?;

    // ---- CUDA driver (subset used by NVENC) ----
    // The CUDA Driver API header ships with the CUDA toolkit (large, version-specific), so we do
    // NOT bundle it. Regenerate cuda.rs from $CUDA_PATH/include/cuda.h when the toolkit is present;
    // otherwise keep the committed binding (the NVENC regen above is self-contained). The 31-symbol
    // subset below is exactly what pixelflux's NVENC path links.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_header = format!("{}/include/cuda.h", cuda_path);
        let cuda_funcs = [
            "cuGetErrorString", "cuGetErrorName", "cuInit", "cuDeviceGetCount", "cuDeviceGet",
            "cuDeviceGetName", "cuDeviceGetUuid", "cuCtxCreate_v2", "cuCtxDestroy_v2",
            "cuCtxPushCurrent_v2", "cuCtxPopCurrent_v2", "cuStreamCreate", "cuStreamDestroy_v2",
            "cuMemAllocHost_v2", "cuMemAllocPitch_v2", "cuMemFree_v2", "cuMemFreeHost",
            "cuMemcpy2D_v2", "cuMemcpy2DUnaligned_v2", "cuMemcpy2DAsync_v2", "cuMemcpyDtoH_v2",
            "cuImportExternalMemory", "cuImportExternalSemaphore", "cuExternalMemoryGetMappedBuffer",
            "cuExternalMemoryGetMappedMipmappedArray", "cuMipmappedArrayGetLevel",
            "cuMipmappedArrayDestroy", "cuDestroyExternalMemory", "cuDestroyExternalSemaphore",
            "cuWaitExternalSemaphoresAsync", "cuSignalExternalSemaphoresAsync",
        ];
        let mut cuda_builder = bindgen::builder()
            .header(&cuda_header)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .size_t_is_usize(true)
            .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
            // Same UB guard: CUresult comes back from the driver (and may postdate this
            // header); CUmemorytype has no zero value yet is zero-filled by Default.
            .newtype_enum("cudaError_enum")
            .newtype_enum("CUmemorytype_enum")
            .generate_comments(false)
            .derive_default(true)
            .derive_eq(true)
            .derive_hash(true)
            .derive_ord(true);
        for f in cuda_funcs {
            cuda_builder = cuda_builder.allowlist_function(f);
        }
        cuda_builder
            .generate()
            .expect("Unable to generate CUDA bindings")
            .write_to_file(out.join("cuda.rs"))
            .expect("Unable to write cuda.rs");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
    } else {
        println!(
            "cargo:warning=CUDA_PATH unset: kept committed src/bindgen/cuda.rs (regenerated NVENC only). Set CUDA_PATH to rebind CUDA."
        );
    }

    println!("cargo:rerun-if-changed=headers/nvEncodeAPI.h");
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}

#[cfg(not(feature = "regen"))]
fn main() {}
