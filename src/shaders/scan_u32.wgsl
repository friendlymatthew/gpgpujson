struct DataBuf {
    data: array<u32>,
}

@group(0)
@binding(0)
var<storage, read_write> global: DataBuf;

var<workgroup> scratch: array<u32, 256>;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    scratch[local_id.x] = global.data[local_id.x];
    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        var stride = 1u << i;
        workgroupBarrier();

        var left = 0u;
        if local_id.x >= stride {
            left = scratch[local_id.x - stride];
        }

        workgroupBarrier();

        scratch[local_id.x] += left;
    }


    global.data[local_id.x] = scratch[local_id.x];
}