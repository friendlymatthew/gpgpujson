struct BufU32 {
    data: array<u32>,
}

struct BufI32 {
    data: array<i32>
}

@group(0)
@binding(0)
var<storage, read_write> depths: BufI32;

@group(0)
@binding(1)
var<storage, read_write> positions: BufU32;

var<workgroup> scratch_depths: array<i32, 256>;
var<workgroup> scratch_positions: array<u32, 256>;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    scratch_depths[local_id.x] = depths.data[local_id.x];
    scratch_positions[local_id.x] = positions.data[local_id.x];
    
    workgroupBarrier();

    for (var k = 2u; k <= 256u; k *= 2u) {
        for (var j = k >> 1u; j > 0u; j >>= 1u) {
            workgroupBarrier();

            let partner = local_id.x ^ j;
            if partner > local_id.x {
                let ascending = (local_id.x & k) == 0u;

                let d0 = scratch_depths[local_id.x];
                let d1 = scratch_depths[partner];
                let p0 = scratch_positions[local_id.x];
                let p1 = scratch_positions[partner];

                let should_swap = (ascending && (d0 > d1 || (d0 == d1 && p0 > p1)))
                               || (!ascending && (d0 < d1 || (d0 == d1 && p0 < p1)));

                if should_swap {
                    scratch_depths[local_id.x] = d1;
                    scratch_depths[partner] = d0;
                    scratch_positions[local_id.x] = p1;
                    scratch_positions[partner] = p0;
                }

            }
        }
    }

    depths.data[local_id.x] = scratch_depths[local_id.x];
    positions.data[local_id.x] = scratch_positions[local_id.x];
}