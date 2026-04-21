use gpgpujson::Gpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TapeEntry {
    byte_pos: u32,
    depth: i32,
    parent: i32,
    char_type: u32,
}

fn parse(gpu: &Gpu, json: &str) -> Vec<TapeEntry> {
    let n = json.len();
    let n_wg = n.div_ceil(256).max(1);
    let padded_len = n_wg * 256;

    let mut input = json.bytes().map(u32::from).collect::<Vec<_>>();
    input.resize(padded_len, 0);

    let input_buf = gpu.storage_buffer("bytes", bytemuck::cast_slice(&input));
    let buf_size = |elem_size: usize| -> u64 { (padded_len * elem_size) as u64 };

    let fsm_buf = gpu.storage_buffer_empty("fsm", buf_size(16));
    gpu.dispatch(
        include_str!("shaders/multi/map_fsm.wgsl"),
        "main",
        &[(&input_buf, true), (&fsm_buf, false)],
        n_wg.try_into().unwrap(),
    );

    gpu.multi_scan_compose(&fsm_buf, padded_len);

    let mask_buf = gpu.storage_buffer_empty("mask", buf_size(4));
    gpu.dispatch(
        include_str!("shaders/multi/map_compact_mask.wgsl"),
        "main",
        &[(&input_buf, true), (&fsm_buf, true), (&mask_buf, false)],
        n_wg.try_into().unwrap(),
    );

    let mask_copy = gpu.storage_buffer_empty("mask_copy", buf_size(4));
    {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&mask_buf, 0, &mask_copy, 0, buf_size(4));
        gpu.queue.submit(Some(encoder.finish()));
    }

    gpu.multi_scan_u32(&mask_buf, padded_len);

    let compact_buf = gpu.storage_buffer_empty("compact", buf_size(4));
    gpu.dispatch(
        include_str!("shaders/multi/scatter_compact.wgsl"),
        "main",
        &[(&mask_copy, true), (&mask_buf, true), (&compact_buf, false)],
        n_wg.try_into().unwrap(),
    );

    let scanned_mask = gpu.read_buffer_as::<u32>(&mask_buf);
    let struct_count = scanned_mask[padded_len - 1] as usize;
    let struct_wg = struct_count.div_ceil(256).max(1);
    let struct_padded = struct_wg * 256;

    let depth_buf = gpu.storage_buffer_empty("depth", (struct_padded * 4) as u64);
    gpu.dispatch(
        include_str!("shaders/multi/map_depth.wgsl"),
        "main",
        &[
            (&input_buf, true),
            (&compact_buf, true),
            (&depth_buf, false),
        ],
        struct_wg.try_into().unwrap(),
    );

    gpu.multi_scan_i32(&depth_buf, struct_padded);

    let parent_buf = gpu.storage_buffer_empty("parents", (struct_padded * 4) as u64);
    gpu.dispatch(
        include_str!("shaders/multi/parent_link.wgsl"),
        "main",
        &[
            (&input_buf, true),
            (&compact_buf, true),
            (&depth_buf, true),
            (&parent_buf, false),
        ],
        struct_wg.try_into().unwrap(),
    );

    let tape_buf = gpu.storage_buffer_empty(
        "tape",
        (struct_padded * std::mem::size_of::<TapeEntry>()) as u64,
    );
    gpu.dispatch(
        include_str!("shaders/multi/assemble_tape.wgsl"),
        "main",
        &[
            (&input_buf, true),
            (&compact_buf, true),
            (&depth_buf, true),
            (&parent_buf, true),
            (&tape_buf, false),
        ],
        struct_wg.try_into().unwrap(),
    );

    let tape = gpu.read_buffer_as::<TapeEntry>(&tape_buf);
    tape[..struct_count].to_vec()
}

fn main() {
    env_logger::init();
    let gpu = Gpu::default();

    let path = std::env::args().nth(1).expect("pass in json as arg");
    let json = std::fs::read_to_string(path).expect("failed");

    let tape = parse(&gpu, &json);
    dbg!(tape);
}
