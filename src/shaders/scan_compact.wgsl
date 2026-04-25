@group(0)
@binding(0)
var<storage, read> global: array<u32>;

@group(0)
@binding(1)
var<storage, read> fsm: array<vec3<u32>>;

@group(0)
@binding(2)
var<storage, read_write> output: array<u32>;

fn read_byte(idx: u32) -> u32 {
    return (global[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

var<workgroup> scratch: array<u32, 256>;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let b = read_byte(local_id.x);
    let is_normal = fsm[local_id.x][0] == 0u;

    var mask = 0u;
    switch b {
        case 0x7Bu, 0x7Du, 0x5Bu, 0x5Du, 0x3Au, 0x2Cu {
            mask = select(0u, 1u, is_normal);
        }
        default {}
    }

    scratch[local_id.x] = mask;

    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        var stride = 1u << i;
        workgroupBarrier();

        var left = 0u;
        if local_id.x >= stride {
            left = scratch[local_id.x - stride];
        }

        workgroupBarrier();

        scratch[local_id.x] += left;
    }

    if mask == 1u {
        output[scratch[local_id.x] - 1u] = local_id.x;
    }
}
