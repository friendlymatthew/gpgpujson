struct BufU32 {
    data: array<u32>,
}

struct BufI32 {
    data: array<i32>,
}

@group(0) @binding(0)
var<storage, read_write> input: BufU32;

@group(0) @binding(1)
var<storage, read_write> compacted: BufU32;

@group(0) @binding(2)
var<storage, read_write> output: BufI32;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;
    let pos = compacted.data[gid];
    let b = input.data[pos];

    var delta: i32 = 0;
    switch b {
        case 0x7Bu, 0x5Bu {
            delta = 1;
        }
        case 0x7Du, 0x5Du {
            delta = -1;
        }
        default {}
    }

    output.data[gid] = delta;
}
