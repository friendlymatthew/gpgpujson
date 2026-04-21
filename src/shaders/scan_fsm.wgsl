struct ReadBuf {
    data: array<u32>,
}

@group(0)
@binding(0)
var<storage, read> global: ReadBuf;

struct OutputBuf {
    data: array<vec3<u32>>,
}

@group(0)
@binding(1)
var<storage, read_write> output: OutputBuf;

var<workgroup> scratch: array<vec3<u32>, 256>;

/*
define 3 states: Normal (0), String (1), Escape (2)

for a given byte, we can't speculate what state its in. so we just compute all possible states
i.e. h -> (0, 1, 1) means:
- if Normal and we encounter `h`, stay Normal
- if String and we encounter `h`, stay String
- if Escape and we encounter `h`, go to String

suppose we need to classify a 3 byte stream `h"w`
their transition functions are:
h -> (0, 1, 1)
" => (1, 0, 1)
w => (0, 1, 1)

compose(h, ") means starting from state S, go through h to get h[S], then feed that into "[h[s]] 
*/
fn compose(lhs: vec3<u32>, rhs: vec3<u32>) -> vec3<u32> {
    return vec3(rhs[lhs[0]], rhs[lhs[1]], rhs[lhs[2]]);
}

const QUOTE = 0x22u;
const ESACPE = 0x5Cu;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {

    switch global.data[local_id.x] {
        case QUOTE {
            scratch[local_id.x] = vec3<u32>(1, 0, 1);
        }
        case ESACPE {
            scratch[local_id.x] = vec3<u32>(0, 2, 1);
        }
        default {
            scratch[local_id.x] = vec3<u32>(0, 1, 1);
        }
    }

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


    output.data[local_id.x] = scratch[local_id.x];
}