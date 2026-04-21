struct BufI32 {
    data: array<i32>,
}

@group(0) @binding(0)
var<storage, read_write> data: BufI32;

@group(0) @binding(1)
var<storage, read_write> totals: BufI32;

var<workgroup> scratch: array<i32, 256>;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    scratch[local_id.x] = data.data[gid];

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

    data.data[gid] = scratch[local_id.x];

    if local_id.x == 255u {
        totals.data[wg_id.x] = scratch[255];
    }
}
