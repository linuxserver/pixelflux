/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Multi-GPU NVENC GET_ATTACHED_IDS / GET_PROBED_IDS ioctl filter.
//!
//! Works across driver versions: on 570-595 (where RM enumeration wrongly returns every host GPU)
//! it drops the unreachable GPUs so the session opens; on 565-or-before / 610-or-later (where
//! enumeration is correct) the strict-subset guard makes it a no-op.
//!
//! On NVIDIA driver 570-595, `libnvidia-encode`/`libcuda`/`libnvcuvid` enumerate every host
//! GPU via the RM `GET_ATTACHED_IDS` ioctl and peer-init each; a GPU whose `/dev/nvidiaX` is
//! absent from the container makes `nvEncOpenEncodeSessionEx` fail with UNSUPPORTED_DEVICE. We
//! GOT-patch `ioctl` in those NVIDIA libraries only (no LD_PRELOAD object; our own GOT is left
//! untouched so the inner real `ioctl` reaches libc) and drop the unreachable GPUs from the
//! response. A strict no-op unless at least one host GPU is hidden from the container.

use libc::{c_char, c_int, c_long, c_ulong, c_void};
use std::ffi::CStr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Once;

#[allow(dead_code)] // NR byte of NV_RM_CONTROL_REQUEST; kept for documentation + the NR unit test.
const NV_ESC_RM_CONTROL: c_ulong = 0x2A; // ioctl NR for NV_ESC_RM_CONTROL
// Canonical ioctl request for NV_ESC_RM_CONTROL: _IOWR('F'=0x46, 0x2A, sizeof(NVOS54_PARAMETERS)=32).
// We match by DIR|TYPE|NR but deliberately IGNORE the encoded _IOC_SIZE (via ioc_no_size), so a
// driver whose NVOS54_PARAMETERS has a different sizeof still matches -- the true per-cmd param
// layout is validated separately by ctrl.params_size below. Matching the NR alone is too loose;
// pinning the exact size is too tight.
const NV_RM_CONTROL_REQUEST: c_ulong = 0xC020_462A;
// _IOC_SIZE field (asm-generic/ioctl.h): a 14-bit size at bit 16. Masked out to compare requests
// size-agnostically.
const IOC_SIZESHIFT: u32 = 16;
const IOC_SIZEMASK: c_ulong = 0x3FFF;
const GPU_GET_ATTACHED_IDS: u32 = 0x0201; // NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
const GPU_GET_PROBED_IDS: u32 = 0x0214; // NV0000_CTRL_CMD_GPU_GET_PROBED_IDS
const MAX_ATTACHED_GPUS: usize = 32;
const INVALID_GPU_ID: u32 = 0xFFFF_FFFF;
// Exact param-struct sizes: ATTACHED = gpuIds[32]; PROBED = gpuIds[32] + excludedGpuIds[32].
// Both begin with gpuIds[32], so we filter that leading array for either; the size disambiguates.
const ATTACHED_PARAMS_SIZE: usize = 4 * MAX_ATTACHED_GPUS;
const PROBED_PARAMS_SIZE: usize = 4 * MAX_ATTACHED_GPUS * 2;

/// NVOS54_PARAMETERS (32 bytes) — the RM control ioctl parameter struct.
#[repr(C)]
struct NvRmControlParams {
    h_client: u32,
    h_object: u32,
    cmd: u32,
    flags: u32,
    params: u64,
    params_size: u32,
    status: u32,
}

// ELF bits libc lacks (Elf64_Sym / Elf64_Phdr / dl_phdr_info / PT_DYNAMIC it has).
const DT_NULL: i64 = 0;
const DT_PLTRELSZ: i64 = 2;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_JMPREL: i64 = 23;

// x86-64 relocation types that name a GOT slot holding a function pointer we can repoint:
// JUMP_SLOT (lazy PLT) and GLOB_DAT (eager / -fno-plt builds).
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;

/// rdev of /dev/nvidiactl, cached once at install() so the ioctl hook only rewrites responses
/// that actually came from that char device (0 = not resolved -> identity gate skipped).
static NVIDIACTL_RDEV: AtomicU64 = AtomicU64::new(0);

#[repr(C)]
struct Elf64Dyn {
    d_tag: i64,
    d_un: u64, // d_val / d_ptr union (both 64-bit)
}

#[repr(C)]
struct Elf64Rela {
    r_offset: u64,
    r_info: u64,
    r_addend: i64,
}

#[inline]
fn elf64_r_sym(info: u64) -> u64 {
    info >> 32
}

#[inline]
fn elf64_r_type(info: u64) -> u32 {
    (info & 0xffff_ffff) as u32
}

#[allow(dead_code)] // NR extractor; kept for documentation + the NR unit test.
#[inline]
fn ioc_nr(req: c_ulong) -> c_ulong {
    // _IOC_NR: bits 0-7 of the ioctl request.
    req & 0xFF
}

/// The ioctl request with its _IOC_SIZE field zeroed, leaving DIR|TYPE|NR. The gate matches on
/// this so it stays bound to the exact RM control command without coupling to one param-struct size.
#[inline]
fn ioc_no_size(req: c_ulong) -> c_ulong {
    req & !(IOC_SIZEMASK << IOC_SIZESHIFT)
}

/// True when `/dev/nvidiaN` exists.
fn node_present(minor: u32) -> bool {
    let path = format!("/dev/nvidia{}\0", minor);
    unsafe { libc::access(path.as_ptr() as *const c_char, libc::F_OK) == 0 }
}

/// Resolve a gpuId to its `/dev/nvidia` minor via /proc (the PCI bus is encoded in
/// `gpuId >> 8`). Returns -1 when no match is found. Matches a /proc entry on either the bus
/// byte alone or the combined domain:bus (`gpuId >> 8`), since larger gpuIds fold the domain in.
fn gpuid_to_minor(gpu_id: u32) -> i32 {
    let want_bus = (gpu_id >> 8) & 0xFF;
    let want_full = gpu_id >> 8;
    let dir = match std::fs::read_dir("/proc/driver/nvidia/gpus") {
        Ok(d) => d,
        Err(_) => return -1,
    };
    for ent in dir.flatten() {
        let name = ent.file_name();
        let name = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        // Parse a PCI address "domain:bus:slot.func" (hex).
        let parts: Vec<&str> = name.split([':', '.']).collect();
        if parts.len() != 4 {
            continue;
        }
        let dom = u32::from_str_radix(parts[0], 16);
        let bus = u32::from_str_radix(parts[1], 16);
        let (dom, bus) = match (dom, bus) {
            (Ok(d), Ok(b)) => (d, b),
            _ => continue,
        };
        if bus != want_bus && ((dom << 8) | bus) != want_full {
            continue;
        }
        // Read "Device Minor: N" from the information file.
        let info = format!("/proc/driver/nvidia/gpus/{}/information", name);
        if let Ok(text) = std::fs::read_to_string(&info) {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("Device Minor:") {
                    if let Ok(m) = rest.trim().parse::<i32>() {
                        return m;
                    }
                }
            }
        }
        break;
    }
    -1
}

/// Pure GET_ATTACHED_IDS rewrite: keep only ids for which `keep(id)` is true, but only when a
/// strict subset survives (`nkept` in `1..total`) — if none or all are kept, leave the array
/// untouched (fail-safe). Factored out so it can be unit-tested without /proc or /dev.
fn filter_ids(ids: &mut [u32; MAX_ATTACHED_GPUS], keep: impl Fn(u32) -> bool) {
    let mut kept = [0u32; MAX_ATTACHED_GPUS];
    let (mut total, mut nkept) = (0usize, 0usize);
    for &id in ids.iter() {
        if id == INVALID_GPU_ID {
            break;
        }
        total += 1;
        if keep(id) {
            kept[nkept] = id;
            nkept += 1;
        }
    }
    if nkept > 0 && nkept < total {
        ids[..nkept].copy_from_slice(&kept[..nkept]);
        for slot in ids[nkept..].iter_mut() {
            *slot = INVALID_GPU_ID;
        }
    }
}

/// ioctl wrapper installed into the NVIDIA libraries' GOT. Our own object's GOT is left
/// untouched, so this inner `libc::ioctl` reaches libc normally (no recursion).
unsafe extern "C" fn filtered_ioctl(fd: c_int, req: c_ulong, arg: *mut c_void) -> c_int {
    let rc = libc::ioctl(fd, req as _, arg);
    // Preserve the real ioctl's errno across our /proc + /dev lookups below (a caller may read
    // errno after the syscall). Also: a panic in the filtering logic must NOT unwind across this
    // extern "C" boundary (the compiler guard would abort the whole process), so catch it and
    // fall back to the unmodified real result.
    let saved_errno = *libc::__errno_location();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rewrite_attached_ids(fd, rc, req, arg);
    }));
    *libc::__errno_location() = saved_errno;
    rc
}

/// In-place rewrite of a successful GET_ATTACHED_IDS / GET_PROBED_IDS response. Split from
/// `filtered_ioctl` so it can run under catch_unwind. Only rewrites when the request is the
/// RM_CONTROL ioctl (matched by DIR|TYPE|NR, size-agnostic) on the real /dev/nvidiactl char device.
unsafe fn rewrite_attached_ids(fd: c_int, rc: c_int, req: c_ulong, arg: *mut c_void) {
    if rc != 0 || ioc_no_size(req) != ioc_no_size(NV_RM_CONTROL_REQUEST) || arg.is_null() {
        return;
    }
    // fd identity: only rewrite responses from the real /dev/nvidiactl char device. Skip the gate
    // (fall back to the request/size checks) only if we never resolved nvidiactl's rdev.
    let cached = NVIDIACTL_RDEV.load(Ordering::Relaxed);
    if cached != 0 {
        let mut st: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut st) != 0 {
            return; // pass through on fstat failure
        }
        if (st.st_mode & libc::S_IFMT) != libc::S_IFCHR || st.st_rdev as u64 != cached {
            return;
        }
    }
    let ctrl = &mut *(arg as *mut NvRmControlParams);
    if ctrl.status != 0 || ctrl.params == 0 {
        return;
    }
    // Filter BOTH enumeration APIs so behavior holds across driver versions: 570-595 (buggy) AND
    // 565-or-before / 610-or-later (correct, where this is a strict-subset no-op). Require the
    // EXACT params size for the cmd so a lying size can't make us rewrite under a different layout.
    let which = match ctrl.cmd {
        GPU_GET_ATTACHED_IDS if ctrl.params_size as usize == ATTACHED_PARAMS_SIZE => "ATTACHED",
        GPU_GET_PROBED_IDS if ctrl.params_size as usize == PROBED_PARAMS_SIZE => "PROBED",
        _ => return,
    };
    // Both params structs start with gpuIds[MAX_ATTACHED_GPUS]; filter that leading array. For
    // PROBED the trailing excludedGpuIds[] is left untouched (it lists GPUs to exclude, not use).
    let ids = &mut *(ctrl.params as *mut [u32; MAX_ATTACHED_GPUS]);
    let debug = std::env::var_os("PIXELFLUX_GPU_FILTER_DEBUG").is_some();
    let before = ids.iter().take_while(|&&id| id != INVALID_GPU_ID).count();
    filter_ids(ids, |id| {
        let minor = gpuid_to_minor(id);
        minor >= 0 && node_present(minor as u32)
    });
    if debug {
        let after = ids.iter().take_while(|&&id| id != INVALID_GPU_ID).count();
        eprintln!("[pixelflux] GET_{which}_IDS intercepted: {before} host GPU(s) -> {after} kept");
    }
}

/// VM protection (PROT_* bits) of the page holding `addr`, from /proc/self/maps (-1 if unknown).
fn page_prot(addr: usize) -> i32 {
    let text = match std::fs::read_to_string("/proc/self/maps") {
        Ok(t) => t,
        Err(_) => return -1,
    };
    for line in text.lines() {
        // "lo-hi perms ..."
        let mut it = line.split_whitespace();
        let range = match it.next() {
            Some(r) => r,
            None => continue,
        };
        let perms = match it.next() {
            Some(p) => p,
            None => continue,
        };
        let mut rr = range.split('-');
        let lo = rr.next().and_then(|s| usize::from_str_radix(s, 16).ok());
        let hi = rr.next().and_then(|s| usize::from_str_radix(s, 16).ok());
        if let (Some(lo), Some(hi)) = (lo, hi) {
            if addr >= lo && addr < hi {
                let b = perms.as_bytes();
                let mut prot = 0;
                if b.first() == Some(&b'r') {
                    prot |= libc::PROT_READ;
                }
                if b.get(1) == Some(&b'w') {
                    prot |= libc::PROT_WRITE;
                }
                if b.get(2) == Some(&b'x') {
                    prot |= libc::PROT_EXEC;
                }
                return prot;
            }
        }
    }
    -1
}

/// Resolve a DT_* d_ptr to an absolute address (glibc rewrites these absolute; musl keeps them
/// file-relative). Heuristic: below base -> relative offset; at/above base -> already absolute.
#[inline]
fn dyn_addr(base: usize, v: u64) -> usize {
    let v = v as usize;
    if v < base {
        base + v
    } else {
        v
    }
}

unsafe fn patch_ioctl_got(base: usize, dynp: *const Elf64Dyn) {
    let mut symtab: *const libc::Elf64_Sym = std::ptr::null();
    let mut strtab: *const c_char = std::ptr::null();
    let mut jmprel: *const Elf64Rela = std::ptr::null();
    let mut pltrelsz: usize = 0;
    let mut rela: *const Elf64Rela = std::ptr::null();
    let mut relasz: usize = 0;

    let mut d = dynp;
    while (*d).d_tag != DT_NULL {
        match (*d).d_tag {
            DT_SYMTAB => symtab = dyn_addr(base, (*d).d_un) as *const libc::Elf64_Sym,
            DT_STRTAB => strtab = dyn_addr(base, (*d).d_un) as *const c_char,
            DT_JMPREL => jmprel = dyn_addr(base, (*d).d_un) as *const Elf64Rela,
            DT_PLTRELSZ => pltrelsz = (*d).d_un as usize,
            DT_RELA => rela = dyn_addr(base, (*d).d_un) as *const Elf64Rela,
            DT_RELASZ => relasz = (*d).d_un as usize,
            _ => {}
        }
        d = d.add(1);
    }
    if symtab.is_null() || strtab.is_null() {
        return;
    }
    // Bail if a resolved table isn't in a readable mapping (a bad relative/absolute guess would
    // otherwise fault on the first dereference).
    let readable = |p: usize| {
        let prot = page_prot(p);
        prot < 0 || (prot & libc::PROT_READ) != 0
    };
    if !readable(symtab as usize) || !readable(strtab as usize) {
        return;
    }

    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as c_long;
    if page <= 0 {
        return;
    }
    let page = page as usize;
    let ent = std::mem::size_of::<Elf64Rela>();
    // PLT relocations (lazy-bound JUMP_SLOT: the classic path).
    if !jmprel.is_null() && pltrelsz >= ent && readable(jmprel as usize) {
        patch_reloc_table(base, symtab, strtab, jmprel, pltrelsz / ent, page);
    }
    // General relocations (GLOB_DAT): NVIDIA libs built with -fno-plt bind `ioctl` through a GOT
    // slot named by an R_X86_64_GLOB_DAT entry in .rela.dyn rather than .rela.plt.
    if !rela.is_null() && relasz >= ent && readable(rela as usize) {
        patch_reloc_table(base, symtab, strtab, rela, relasz / ent, page);
    }
}

/// Scan one relocation table and repoint every GOT slot naming `ioctl` (JUMP_SLOT or GLOB_DAT)
/// to `filtered_ioctl`.
unsafe fn patch_reloc_table(
    base: usize,
    symtab: *const libc::Elf64_Sym,
    strtab: *const c_char,
    rela: *const Elf64Rela,
    count: usize,
    page: usize,
) {
    for i in 0..count {
        let r = rela.add(i);
        let rtype = elf64_r_type((*r).r_info);
        if rtype != R_X86_64_JUMP_SLOT && rtype != R_X86_64_GLOB_DAT {
            continue;
        }
        let sym_idx = elf64_r_sym((*r).r_info) as usize;
        if sym_idx == 0 {
            continue;
        }
        let name_off = (*symtab.add(sym_idx)).st_name as usize;
        let name = CStr::from_ptr(strtab.add(name_off));
        if name.to_bytes() != b"ioctl" {
            continue;
        }
        let slot = (base + (*r).r_offset as usize) as *mut *mut c_void;
        let pg = (slot as usize & !(page - 1)) as *mut c_void;
        // Restore the slot's original protection (writable under partial RELRO) rather than a
        // hardcoded read-only: these libs lazily bind through this page, so read-only faults the
        // next resolve. Default to writable if maps is unreadable.
        let mut orig = page_prot(slot as usize);
        if orig < 0 {
            orig = libc::PROT_READ | libc::PROT_WRITE;
        }
        if libc::mprotect(pg, page, libc::PROT_READ | libc::PROT_WRITE) == 0 {
            // Atomic pointer store: another thread may be dispatching through this GOT slot
            // concurrently, so publish the new pointer without tearing.
            let ap = &*(slot as *const AtomicPtr<c_void>);
            ap.store(filtered_ioctl as *mut c_void, Ordering::Release);
            libc::mprotect(pg, page, orig);
        }
    }
}

unsafe extern "C" fn patch_phdr_cb(
    info: *mut libc::dl_phdr_info,
    _size: libc::size_t,
    _data: *mut c_void,
) -> c_int {
    // A panic must not unwind across this extern "C" boundary (dl_iterate_phdr is C; the guard
    // would abort the process): catch it and keep iterating.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let info = &*info;
        if info.dlpi_name.is_null() || *info.dlpi_name == 0 {
            return;
        }
        let name = CStr::from_ptr(info.dlpi_name).to_string_lossy();
        // GET_ATTACHED_IDS is issued by libcuda and libnvcuvid (libnvidia-encode calls through
        // libnvcuvid), so exactly those three are patched. Match tightly so unrelated libraries
        // (libcudart, libnvidia-ml, libnvidia-glcore, ...) are left untouched.
        if !name.contains("libnvcuvid")
            && !name.contains("libnvidia-encode")
            && !name.contains("libcuda.so")
        {
            return;
        }
        let base = info.dlpi_addr as usize;
        for i in 0..info.dlpi_phnum as isize {
            let ph = &*info.dlpi_phdr.offset(i);
            if ph.p_type == libc::PT_DYNAMIC {
                patch_ioctl_got(base, (base + ph.p_vaddr as usize) as *const Elf64Dyn);
            }
        }
    }));
    0
}

/// True when at least one host GPU is hidden from the container (the only case the peer-init bug
/// can trigger), so the filter is a no-op everywhere else.
fn has_hidden_gpus() -> bool {
    let host = std::fs::read_dir("/proc/driver/nvidia/gpus")
        .map(|d| d.flatten().filter(|e| !e.file_name().to_string_lossy().starts_with('.')).count())
        .unwrap_or(0);
    let visible = (0..MAX_ATTACHED_GPUS as u32).filter(|&m| node_present(m)).count();
    host > visible && visible > 0
}

/// Install the GET_ATTACHED_IDS GOT filter once, but only when a host GPU is hidden from the
/// container. Safe to call before every NVENC session open; idempotent.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Escape hatch to skip the GOT patch entirely (also handy for A/B testing it).
        if std::env::var_os("PIXELFLUX_DISABLE_GPU_FILTER").is_some() {
            eprintln!("[pixelflux] multi-GPU NVENC filter disabled via PIXELFLUX_DISABLE_GPU_FILTER");
            return;
        }
        // Cache /dev/nvidiactl's rdev so the ioctl hook can verify the fd identity before it ever
        // rewrites a response (0 stays cached on failure -> the identity gate is skipped).
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            if libc::stat(c"/dev/nvidiactl".as_ptr(), &mut st) == 0 {
                NVIDIACTL_RDEV.store(st.st_rdev as u64, Ordering::Relaxed);
            }
        }
        if has_hidden_gpus() {
            unsafe {
                libc::dl_iterate_phdr(Some(patch_phdr_cb), std::ptr::null_mut());
            }
            eprintln!("[pixelflux] multi-GPU NVENC ioctl filter installed (GET_ATTACHED_IDS/GET_PROBED_IDS)");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_keeps_strict_subset_and_invalidates_rest() {
        let mut ids = [INVALID_GPU_ID; MAX_ATTACHED_GPUS];
        ids[0] = 0x100;
        ids[1] = 0x200; // dropped
        ids[2] = 0x300;
        // total=3, keep 0x100 & 0x300 -> strict subset (2<3) => rewrite.
        filter_ids(&mut ids, |id| id == 0x100 || id == 0x300);
        assert_eq!(ids[0], 0x100);
        assert_eq!(ids[1], 0x300);
        assert_eq!(ids[2], INVALID_GPU_ID);
        assert_eq!(ids[3], INVALID_GPU_ID);
    }

    #[test]
    fn filter_noop_when_all_kept() {
        let mut ids = [INVALID_GPU_ID; MAX_ATTACHED_GPUS];
        ids[0] = 0xAA;
        ids[1] = 0xBB;
        filter_ids(&mut ids, |_| true); // nkept==total => no rewrite
        assert_eq!(ids[0], 0xAA);
        assert_eq!(ids[1], 0xBB);
        assert_eq!(ids[2], INVALID_GPU_ID);
    }

    #[test]
    fn filter_noop_when_none_kept() {
        let mut ids = [INVALID_GPU_ID; MAX_ATTACHED_GPUS];
        ids[0] = 0xAA;
        ids[1] = 0xBB;
        filter_ids(&mut ids, |_| false); // nkept==0 => fail-safe, leave untouched
        assert_eq!(ids[0], 0xAA);
        assert_eq!(ids[1], 0xBB);
    }

    #[test]
    fn ioc_nr_extracts_low_byte() {
        assert_eq!(ioc_nr(0xC020462A), NV_ESC_RM_CONTROL);
    }

    #[test]
    fn request_match_ignores_param_size() {
        // A request sharing DIR|TYPE|NR but encoding a DIFFERENT _IOC_SIZE must still match, so a
        // driver whose NVOS54_PARAMETERS has a different sizeof isn't rejected by the request gate.
        let base = ioc_no_size(NV_RM_CONTROL_REQUEST);
        let other_size = base | (0x30 << IOC_SIZESHIFT); // size 0x30 instead of the canonical 0x20
        assert_ne!(other_size, NV_RM_CONTROL_REQUEST);
        assert_eq!(ioc_no_size(other_size), base);
        // NR precision is preserved: a different NR does NOT match.
        let other_nr = (NV_RM_CONTROL_REQUEST & !0xFF) | 0x2B;
        assert_ne!(ioc_no_size(other_nr), base);
    }

    #[test]
    fn param_struct_sizes_match_nvidia_layout() {
        // ATTACHED = gpuIds[32]; PROBED = gpuIds[32] + excludedGpuIds[32]. Both lead with the
        // gpuIds[32] we rewrite, so the exact-size guards disambiguate the two cmds.
        assert_eq!(ATTACHED_PARAMS_SIZE, 128);
        assert_eq!(PROBED_PARAMS_SIZE, 256);
        assert_eq!(std::mem::size_of::<NvRmControlParams>(), 32);
    }
}
