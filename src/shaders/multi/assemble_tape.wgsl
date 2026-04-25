struct TapeEntry {
    byte_pos: u32,
    depth: i32,
    parent: i32,
    char_type: u32,
}

@group(0) @binding(0)
var<storage, read> global: array<u32>;

@group(0) @binding(1)
var<storage, read> compacted: array<u32>;

fn read_byte(idx: u32) -> u32 {
    return (global[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

@group(0) @binding(2)
var<storage, read> depths: array<i32>;

@group(0) @binding(3)
var<storage, read> parents: array<i32>;

@group(0) @binding(4)
var<storage, read_write> tape: array<TapeEntry>;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;
    let pos = compacted[gid];

    tape[gid] = TapeEntry(
        pos,
        depths[gid],
        parents[gid],
        read_byte(pos),
    );
}
