@group(0) @binding(0)
var<storage, read> input: array<u32>;

@group(0) @binding(1)
var<storage, read> fsm: array<vec3<u32>>;

@group(0) @binding(2)
var<storage, read_write> mask: array<u32>;

fn read_byte(idx: u32) -> u32 {
    return (input[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    let b = read_byte(gid);
    let is_normal = fsm[gid][0] == 0u;

    var m = 0u;
    switch b {
        case 0x7Bu, 0x7Du, 0x5Bu, 0x5Du, 0x3Au, 0x2Cu {
            m = select(0u, 1u, is_normal);
        }
        default {}
    }

    mask[gid] = m;
}
