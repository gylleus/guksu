//! Minimal matrix loader: .npy (v1/v2 header, `<f4`, C-order, 2-D) or raw
//! little-endian f32 with an explicit `--dim`. Files are format-sniffed by the
//! `\x93NUMPY` magic, not by extension.

use std::path::Path;

/// Load a matrix; returns (flat row-major f32, dim). `raw_dim` is required for
/// raw (non-.npy) files.
pub fn load_matrix(path: &Path, raw_dim: Option<usize>) -> Result<(Vec<f32>, usize), String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.starts_with(b"\x93NUMPY") {
        parse_npy(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    } else {
        let dim = raw_dim.ok_or_else(|| {
            format!("{} is not .npy; raw f32 files need --dim", path.display())
        })?;
        if bytes.len() % (4 * dim) != 0 {
            return Err(format!(
                "{}: {} bytes is not whole rows of dim {dim} (row = {} bytes)",
                path.display(),
                bytes.len(),
                4 * dim
            ));
        }
        Ok((le_f32(&bytes), dim))
    }
}

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn parse_npy(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
    if bytes.len() < 10 {
        return Err("truncated .npy header".into());
    }
    let (major, _minor) = (bytes[6], bytes[7]);
    let (header, data_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (bytes.get(10..10 + len).ok_or("truncated v1 header")?, 10 + len)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated v2 header".into());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (bytes.get(12..12 + len).ok_or("truncated v2 header")?, 12 + len)
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(header).map_err(|_| "non-utf8 .npy header")?;

    let descr = dict_str(header, "descr").ok_or("missing 'descr' in .npy header")?;
    if descr != "<f4" {
        return Err(format!("expected descr '<f4' (little-endian f32), got {descr:?}"));
    }
    match dict_raw(header, "fortran_order").ok_or("missing 'fortran_order'")? {
        "False" => {}
        "True" => return Err("fortran_order: True is not supported (save C-order)".into()),
        other => return Err(format!("unparseable fortran_order {other:?}")),
    }
    let shape = dict_raw(header, "shape").ok_or("missing 'shape'")?;
    let dims = parse_shape(shape)?;
    let [n, dim] = dims[..] else {
        return Err(format!(
            "expected 2-D shape (n, dim), got {shape} — reshape to (n, dim) before saving"
        ));
    };
    if dim == 0 {
        return Err("dim must be > 0".into());
    }

    let data = &bytes[data_start..];
    if data.len() != n * dim * 4 {
        return Err(format!(
            "data section is {} bytes, expected {} for shape ({n}, {dim}) f32",
            data.len(),
            n * dim * 4
        ));
    }
    Ok((le_f32(data), dim))
}

/// Value of a quoted-string dict entry, tolerant of key order and whitespace.
fn dict_str(header: &str, key: &str) -> Option<String> {
    let raw = dict_raw(header, key)?;
    let raw = raw.trim();
    if raw.len() >= 2 && (raw.starts_with('\'') || raw.starts_with('"')) {
        Some(raw[1..raw.len() - 1].to_string())
    } else {
        None
    }
}

/// Raw text of a dict entry value up to the next top-level ',' or '}'.
fn dict_raw<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    for quote in ['\'', '"'] {
        let pat = format!("{quote}{key}{quote}");
        if let Some(kpos) = header.find(&pat) {
            let after = &header[kpos + pat.len()..];
            let colon = after.find(':')?;
            let value = after[colon + 1..].trim_start();
            let mut depth = 0usize;
            for (i, ch) in value.char_indices() {
                match ch {
                    '(' | '[' => depth += 1,
                    ')' | ']' if depth > 0 => depth -= 1,
                    ',' | '}' if depth == 0 => return Some(value[..i].trim()),
                    _ => {}
                }
            }
            return Some(value.trim_end().trim_end_matches('}').trim());
        }
    }
    None
}

fn parse_shape(shape: &str) -> Result<Vec<usize>, String> {
    let inner = shape
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("unparseable shape {shape:?}"))?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| format!("bad shape element {s:?}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid .npy byte vec in memory (v1 or v2).
    fn npy(version: u8, header: &str, data: &[f32]) -> Vec<u8> {
        let mut out = b"\x93NUMPY".to_vec();
        out.push(version);
        out.push(0);
        let h = header.as_bytes();
        match version {
            1 => out.extend((h.len() as u16).to_le_bytes()),
            _ => out.extend((h.len() as u32).to_le_bytes()),
        }
        out.extend(h);
        for x in data {
            out.extend(x.to_le_bytes());
        }
        out
    }

    fn load(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        parse_npy(bytes)
    }

    const DATA: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    #[test]
    fn parses_v1_and_v2_headers() {
        for v in [1u8, 2] {
            let bytes =
                npy(v, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }      \n", &DATA);
            let (flat, dim) = load(&bytes).unwrap();
            assert_eq!(dim, 3);
            assert_eq!(flat, DATA);
        }
    }

    #[test]
    fn tolerates_key_order_and_whitespace() {
        let bytes = npy(1, "{ 'shape':(2,3),'fortran_order' : False , 'descr':'<f4' }", &DATA);
        assert_eq!(load(&bytes).unwrap().1, 3);
        let dq = npy(1, "{\"descr\": \"<f4\", \"fortran_order\": False, \"shape\": (2, 3)}", &DATA);
        assert_eq!(load(&dq).unwrap().1, 3);
    }

    #[test]
    fn rejects_wrong_dtype_order_and_shape() {
        let be = npy(1, "{'descr': '>f4', 'fortran_order': False, 'shape': (2, 3), }", &DATA);
        assert!(load(&be).unwrap_err().contains("<f4"));
        let f8 = npy(1, "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }", &DATA);
        assert!(load(&f8).unwrap_err().contains("<f4"));
        let fortran = npy(1, "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }", &DATA);
        assert!(load(&fortran).unwrap_err().contains("C-order"));
        let one_d = npy(1, "{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }", &DATA);
        assert!(load(&one_d).unwrap_err().contains("reshape"));
        let short = npy(1, "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 3), }", &DATA);
        assert!(load(&short).unwrap_err().contains("expected"));
    }

    #[test]
    fn raw_loader_needs_dim_and_whole_rows() {
        let dir = std::env::temp_dir().join("guksu_npy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("raw.f32");
        let bytes: Vec<u8> = DATA.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_matrix(&path, None).unwrap_err().contains("--dim"));
        assert!(load_matrix(&path, Some(4)).unwrap_err().contains("whole rows"));
        let (flat, dim) = load_matrix(&path, Some(3)).unwrap();
        assert_eq!((flat, dim), (DATA.to_vec(), 3));
    }
}
