@group(0) @binding(0)
var<storage, read> mask: array<u32>;

struct Indirect {
    x: u32,
    y: u32,
    z: u32,
}

@group(0) @binding(1)
var<storage, read_write> indirect: Indirect;

@compute
@workgroup_size(1)
fn main() {
    let n = arrayLength(&mask);
    let struct_count = mask[n - 1u];
    indirect.x = (struct_count + 255u) / 256u;
    indirect.y = 1u;
    indirect.z = 1u;
}
