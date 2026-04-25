@group(0) @binding(0)
var<storage, read> input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<vec3<u32>>;

const QUOTE = 0x22u;
const ESCAPE = 0x5Cu;

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

    switch read_byte(gid) {
        case QUOTE {
            output[gid] = vec3<u32>(1, 0, 1);
        }
        case ESCAPE {
            output[gid] = vec3<u32>(0, 2, 1);
        }
        default {
            output[gid] = vec3<u32>(0, 1, 1);
        }
    }
}
