@group(0) @binding(0)
var<storage, read> mask: array<u32>;

@group(0) @binding(1)
var<storage, read> scanned: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    if mask[gid] == 1u {
        let out_idx = scanned[gid] - 1u;
        output[out_idx] = gid;
    }
}
