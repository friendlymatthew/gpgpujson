
struct FsmBuf {
    data: array<vec3<u32>>,
}

@group(0) @binding(0)
var<storage, read_write> output: FsmBuf;

@group(0) @binding(1)
var<storage, read_write> totals: FsmBuf;

fn compose(lhs: vec3<u32>, rhs: vec3<u32>) -> vec3<u32> {
    return vec3(rhs[lhs[0]], rhs[lhs[1]], rhs[lhs[2]]);
}

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
    let prefix = totals.data[wg_id.x - 1u];

    output.data[gid] = compose(prefix, output.data[gid]);
}
