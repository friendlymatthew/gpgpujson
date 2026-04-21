struct BufU32 {
    data: array<u32>,
}

struct FsmBuf {
    data: array<vec3<u32>>,
}

@group(0) @binding(0)
var<storage, read_write> input: BufU32;

@group(0) @binding(1)
var<storage, read_write> fsm: FsmBuf;

@group(0) @binding(2)
var<storage, read_write> mask: BufU32;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    let b = input.data[gid];
    let is_normal = fsm.data[gid][0] == 0u;

    var m = 0u;
    switch b {
        case 0x7Bu, 0x7Du, 0x5Bu, 0x5Du, 0x3Au, 0x2Cu {
            m = select(0u, 1u, is_normal);
        }
        default {}
    }

    mask.data[gid] = m;
}
