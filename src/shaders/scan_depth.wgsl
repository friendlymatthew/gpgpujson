@group(0)
@binding(0)
var<storage, read> global: array<u32>;

@group(0)
@binding(1)
var<storage, read> compacted: array<u32>;

fn read_byte(idx: u32) -> u32 {
    return (global[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

@group(0)
@binding(2)
var<storage, read_write> output: array<i32>;

var<workgroup> scratch: array<i32, 256>;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let i = compacted[local_id.x];
    let b = read_byte(i);

    var depth: i32 = 0;

    switch b {
        case 0x7Bu, 0x5Bu {
            depth += 1;
        }
        case 0x7Du, 0x5Du {
            depth -= 1;
        }
        default {}
    }

    scratch[local_id.x] = depth;

    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        var stride = 1u << i;
        workgroupBarrier();

        var left = 0i;
        if local_id.x >= stride {
            left = scratch[local_id.x - stride];
        }

        workgroupBarrier();

        scratch[local_id.x] += left;
    }

    output[local_id.x] = scratch[local_id.x];
}
