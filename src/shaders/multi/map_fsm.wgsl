struct BufU32 {
    data: array<u32>,
}

struct FsmBuf {
    data: array<vec3<u32>>,
}

@group(0) @binding(0)
var<storage, read> input: BufU32;

@group(0) @binding(1)
var<storage, read_write> output: FsmBuf;

const QUOTE = 0x22u;
const ESCAPE = 0x5Cu;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let gid = wg_id.x * 256u + local_id.x;

    switch input.data[gid] {
        case QUOTE {
            output.data[gid] = vec3<u32>(1, 0, 1);
        }
        case ESCAPE {
            output.data[gid] = vec3<u32>(0, 2, 1);
        }
        default {
            output.data[gid] = vec3<u32>(0, 1, 1);
        }
    }
}
