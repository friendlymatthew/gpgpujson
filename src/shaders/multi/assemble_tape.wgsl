struct BufU32 {
    data: array<u32>,
}

struct BufI32 {
    data: array<i32>,
}

struct TapeEntry {
    byte_pos: u32,
    depth: i32,
    parent: i32,
    char_type: u32,
}

struct TapeBuf {
    data: array<TapeEntry>,
}

@group(0) @binding(0)
var<storage, read> global: BufU32;

@group(0) @binding(1)
var<storage, read> compacted: BufU32;

@group(0) @binding(2)
var<storage, read> depths: BufI32;

@group(0) @binding(3)
var<storage, read> parents: BufI32;

@group(0) @binding(4)
var<storage, read_write> tape: TapeBuf;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;
    let pos = compacted.data[gid];

    tape.data[gid] = TapeEntry(
        pos,
        depths.data[gid],
        parents.data[gid],
        global.data[pos],
    );
}
