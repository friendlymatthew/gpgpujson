struct BufU32 {
    data: array<u32>,
}

struct BufI32 {
    data: array<i32>,
}

@group(0) @binding(0)
var<storage, read_write> global: BufU32;

@group(0) @binding(1)
var<storage, read_write> compacted: BufU32;

@group(0) @binding(2)
var<storage, read_write> depths: BufI32;

@group(0) @binding(3)
var<storage, read_write> parents: BufI32;

const LBRACE = 0x7Bu;
const RBRACE = 0x7Du;
const LBRACKET = 0x5Bu;
const RBRACKET = 0x5Du;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    let my_depth = depths.data[gid];
    let my_byte = global.data[compacted.data[gid]];

    var target_depth: i32;
    if my_byte == LBRACE || my_byte == LBRACKET {
        target_depth = my_depth - 1;
    } else if my_byte == RBRACE || my_byte == RBRACKET {
        target_depth = my_depth + 1;
    } else {
        target_depth = my_depth;
    }

    if target_depth <= 0 {
        parents.data[gid] = -1;
        return;
    }

    var parent = -1;
    for (var j = i32(gid) - 1; j >= 0; j--) {
        let d = depths.data[u32(j)];
        if d == target_depth {
            let b = global.data[compacted.data[u32(j)]];
            if b == LBRACE || b == LBRACKET {
                parent = j;
                break;
            }
        }
    }

    parents.data[gid] = parent;
}
