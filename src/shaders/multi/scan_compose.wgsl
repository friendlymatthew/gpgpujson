struct FsmBuf {
    data: array<vec3<u32>>,
}

@group(0) @binding(0)
var<storage, read_write> data: FsmBuf;

var<workgroup> scratch: array<vec3<u32>, 256>;

fn compose(lhs: vec3<u32>, rhs: vec3<u32>) -> vec3<u32> {
    return vec3(rhs[lhs[0]], rhs[lhs[1]], rhs[lhs[2]]);
}

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    scratch[local_id.x] = data.data[local_id.x];

    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        var stride = 1u << i;
        workgroupBarrier();

        var left = vec3<u32>(0, 1, 2);
        if local_id.x >= stride {
            left = scratch[local_id.x - stride];
        }

        workgroupBarrier();

        scratch[local_id.x] = compose(left, scratch[local_id.x]);
    }

    data.data[local_id.x] = scratch[local_id.x];
}
