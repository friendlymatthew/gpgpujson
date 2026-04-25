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
var<storage, read> depths: array<i32>;

@group(0)
@binding(3)
var<storage, read_write> parents: array<i32>;

const LBRACE = 0x7Bu;
const RBRACE = 0x7Du;
const LBRACKET = 0x5Bu;
const RBRACKET = 0x5Du;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let my_depth = depths[local_id.x];
    let my_byte = read_byte(compacted[local_id.x]);

    // determine target depth based on character type:
    // openers ({, [) at depth d: parent = opener at d - 1
    // closers (}, ]) at depth d: parent = opener at d + 1
    // others  (:, ,) at depth d: parent = opener at d
    var target_depth: i32;
    if my_byte == LBRACE || my_byte == LBRACKET {
        target_depth = my_depth - 1;
    } else if my_byte == RBRACE || my_byte == RBRACKET {
        target_depth = my_depth + 1;
    } else {
        target_depth = my_depth;
    }

    if target_depth <= 0 {
        parents[local_id.x] = -1;
        return;
    }

    var parent = -1;
    for (var j = i32(local_id.x) - 1; j >= 0; j--) {
        let d = depths[u32(j)];
        if d == target_depth {
            let b = read_byte(compacted[u32(j)]);
            if b == LBRACE || b == LBRACKET {
                parent = j;
                break;
            }
        }
    }

    parents[local_id.x] = parent;
}
