struct BufI32 {
    data: array<i32>,
}

@group(0) @binding(0)
var<storage, read_write> data: BufI32;

@group(0) @binding(1)
var<storage, read> totals: BufI32;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    if wg_id.x == 0u {
        return;
    }

    let gid = wg_id.x * 256u + local_id.x;
    data.data[gid] += totals.data[wg_id.x - 1u];
}
