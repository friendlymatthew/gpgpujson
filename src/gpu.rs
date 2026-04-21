use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpu {
    pub fn try_new() -> Result<Self> {
        pollster::block_on(Self::init())
    }

    pub fn new() -> Self {
        Self::try_new().expect("failed to initialize GPU")
    }

    async fn init() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow!("no suitable GPU adapter found"))?;

        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok(Self { device, queue })
    }

    pub fn storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn storage_buffer_empty(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn read_buffer(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let staging = self.staging_buffer(buffer.size());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        data.to_vec()
    }

    pub fn read_buffer_as<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let bytes = self.read_buffer(buffer);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    pub fn dispatch(
        &self,
        shader_src: &str,
        entry_point: &str,
        buffers: &[(&wgpu::Buffer, bool)],
        workgroups: u32,
    ) {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let bind_group_layout_entries = buffers
            .iter()
            .enumerate()
            .map(|(i, &(_, read_only))| wgpu::BindGroupLayoutEntry {
                binding: i.try_into().unwrap(),
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let bind_group_entries = buffers
            .iter()
            .enumerate()
            .map(|(i, (buf, _))| wgpu::BindGroupEntry {
                binding: i.try_into().unwrap(),
                resource: buf.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 2);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn multi_scan_u32(&self, buf: &wgpu::Buffer, n: usize) {
        self.multi_scan_generic(
            buf,
            n,
            4,
            include_str!("shaders/multi/scan_u32_local.wgsl"),
            include_str!("shaders/scan_u32.wgsl"),
            include_str!("shaders/multi/propagate_u32.wgsl"),
        );
    }

    pub fn multi_scan_i32(&self, buf: &wgpu::Buffer, n: usize) {
        self.multi_scan_generic(
            buf,
            n,
            4,
            include_str!("shaders/multi/scan_i32_local.wgsl"),
            include_str!("shaders/multi/scan_i32.wgsl"),
            include_str!("shaders/multi/propagate_i32.wgsl"),
        );
    }

    pub fn multi_scan_fsm(&self, buf: &wgpu::Buffer, n: usize) {
        self.multi_scan_generic(
            buf,
            n,
            16,
            include_str!("shaders/multi/scan_compose_local.wgsl"),
            include_str!("shaders/multi/scan_compose.wgsl"),
            include_str!("shaders/multi/fsm_propagate.wgsl"),
        );
    }

    fn multi_scan_generic(
        &self,
        buf: &wgpu::Buffer,
        n: usize,
        elem_size: usize,
        // the shader that scans within each workgroup
        local_shader: &str,
        // the shader that scans a buffer that fits in 1 workgroup (base case)
        single_shader: &str,
        // adds each workgroup's prefix back down (data + totals)
        propagate_shader: &str,
    ) {
        if n <= 256 {
            self.dispatch(single_shader, "main", &[(buf, false)], 1);
            return;
        }

        let level_sizes = std::iter::successors(Some(n), |&s| (s > 256).then(|| s.div_ceil(256)))
            .collect::<Vec<_>>();

        let totals = (1..level_sizes.len())
            .map(|i| {
                let size = level_sizes[i].max(256);
                self.storage_buffer_empty("totals", (size * elem_size) as u64)
            })
            .collect::<Vec<_>>();

        for i in 0..totals.len() {
            let data = if i == 0 { buf } else { &totals[i - 1] };
            let n_wg = level_sizes[i + 1];

            if n_wg == 1 {
                self.dispatch(single_shader, "main", &[(data, false)], 1);
            } else {
                self.dispatch(
                    local_shader,
                    "main",
                    &[(data, false), (&totals[i], false)],
                    n_wg.try_into().unwrap(),
                );
            }
        }

        self.dispatch(single_shader, "main", &[(totals.last().unwrap(), false)], 1);

        for i in (0..totals.len()).rev() {
            let data = if i == 0 { buf } else { &totals[i - 1] };
            let n_wg = level_sizes[i + 1];
            self.dispatch(
                propagate_shader,
                "main",
                &[(data, false), (&totals[i], true)],
                n_wg.try_into().unwrap(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_u32() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let input = vec![1u32; 256];
        let buf = gpu.storage_buffer("scan", bytemuck::cast_slice(&input));

        gpu.dispatch(
            include_str!("shaders/scan_u32.wgsl"),
            "main",
            &[(&buf, false)],
            1 << 4,
        );

        let out = gpu.read_buffer_as::<u32>(&buf);
        assert_eq!(out, (1..=256).collect::<Vec<_>>());
    }

    #[test]
    fn test_fsm_string_detect() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };

        let input_str = r#"hello "world" end"#;

        let mut input = input_str.bytes().map(u32::from).collect::<Vec<_>>();
        input.resize(256, 0);

        let input_buf = gpu.storage_buffer("fsm_in", bytemuck::cast_slice(&input));
        let output_buf = gpu.storage_buffer_empty(
            "fsm_out",
            // vec3 is padded to 16 bytes in storage
            (256 * std::mem::size_of::<[u32; 4]>()) as u64,
        );

        gpu.dispatch(
            include_str!("shaders/scan_fsm.wgsl"),
            "main",
            &[(&input_buf, true), (&output_buf, false)],
            1,
        );

        let out = gpu.read_buffer_as::<[u32; 4]>(&output_buf);

        let states = out[..input_str.len()]
            .iter()
            .map(|f| f[0])
            .collect::<Vec<_>>();

        //                  h  e  l  l  o     "  w  o  r  l  d  "     e  n  d
        assert_eq!(states, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_fsm_escape() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };

        // the \" inside the string should NOT close it
        let input_str = r#"hello "wo\"rld" end"#;

        let mut input = input_str.bytes().map(u32::from).collect::<Vec<_>>();
        input.resize(256, 0);

        let input_buf = gpu.storage_buffer("fsm_in", bytemuck::cast_slice(&input));
        let output_buf =
            gpu.storage_buffer_empty("fsm_out", (256 * std::mem::size_of::<[u32; 4]>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_fsm.wgsl"),
            "main",
            &[(&input_buf, true), (&output_buf, false)],
            1,
        );

        let out = gpu.read_buffer_as::<[u32; 4]>(&output_buf);

        let states = out[..input_str.len()]
            .iter()
            .map(|f| f[0])
            .collect::<Vec<_>>();

        assert_eq!(
            states,
            // h   e  l  l  o     "  w  o  \  "  r  l  d  "     e  n  d
            [0u32, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_depth() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };

        // {"a":{"b":[1,2]}}
        let input_str = r#"{"a":{"b":[1,2]}}"#;

        let mut input = input_str.bytes().map(u32::from).collect::<Vec<_>>();
        input.resize(256, 0);

        let input_buf = gpu.storage_buffer("bytes", bytemuck::cast_slice(&input));
        let fsm_buf =
            gpu.storage_buffer_empty("fsm", (256 * std::mem::size_of::<[u32; 4]>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_fsm.wgsl"),
            "main",
            &[(&input_buf, true), (&fsm_buf, false)],
            1,
        );

        let compact_buf =
            gpu.storage_buffer_empty("compact", (256 * std::mem::size_of::<u32>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_compact.wgsl"),
            "main",
            &[(&input_buf, true), (&fsm_buf, true), (&compact_buf, false)],
            1,
        );

        let depth_buf =
            gpu.storage_buffer_empty("depth", (256 * std::mem::size_of::<i32>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_depth.wgsl"),
            "main",
            &[
                (&input_buf, true),
                (&compact_buf, true),
                (&depth_buf, false),
            ],
            1,
        );

        let depths = gpu.read_buffer_as::<i32>(&depth_buf);

        let mut state = 0u32;
        let mut depth = 0i32;
        let mut expected = vec![];
        for b in input_str.bytes() {
            state = match (state, b) {
                (0, b'"') => 1,
                (1, b'"') => 0,
                (1, b'\\') => 2,
                (2, _) => 1,
                (s, _) => s,
            };

            if state == 0 && matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b',') {
                match b {
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => depth -= 1,
                    _ => {}
                }
                expected.push(depth);
            }
        }

        let count = expected.len();
        assert_eq!(&depths[..count], &expected);
    }

    #[test]
    fn test_sort() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };

        // compacted structural chars from {"a":{"b":[1,2]}}
        // positions: {  :  {  :  [  ,  ]  }  }
        //            0  4  5  9  10 12 14 15 17
        // depths:    1  1  2  2  3  3  2  1  0
        let mut positions: Vec<u32> = vec![0, 4, 5, 9, 10, 12, 14, 15, 17];
        let mut depths: Vec<i32> = vec![1, 1, 2, 2, 3, 3, 2, 1, 0];

        positions.resize(256, 0);
        depths.resize(256, 0);

        let depth_buf = gpu.storage_buffer("depths", bytemuck::cast_slice(&depths));
        let pos_buf = gpu.storage_buffer("positions", bytemuck::cast_slice(&positions));

        gpu.dispatch(
            include_str!("shaders/sort.wgsl"),
            "main",
            &[(&depth_buf, false), (&pos_buf, false)],
            1,
        );

        let sorted_depths = gpu.read_buffer_as::<i32>(&depth_buf);
        let sorted_positions = gpu.read_buffer_as::<u32>(&pos_buf);

        let mut pairs = depths[..256]
            .iter()
            .zip(positions[..256].iter())
            .map(|(&d, &p)| (d, p))
            .collect::<Vec<_>>();

        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let expected_depths = pairs.iter().map(|p| p.0).collect::<Vec<_>>();
        let expected_positions = pairs.iter().map(|p| p.1).collect::<Vec<_>>();

        assert_eq!(sorted_depths, expected_depths);
        assert_eq!(sorted_positions, expected_positions);
    }

    #[test]
    fn test_parent_link() {
        let gpu = match Gpu::try_new() {
            Ok(g) => g,
            Err(_) => return,
        };

        let input_str = r#"{"a":{"b":[1,2]}}"#;

        let mut input = input_str.bytes().map(u32::from).collect::<Vec<_>>();
        input.resize(256, 0);

        let input_buf = gpu.storage_buffer("bytes", bytemuck::cast_slice(&input));
        let fsm_buf =
            gpu.storage_buffer_empty("fsm", (256 * std::mem::size_of::<[u32; 4]>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_fsm.wgsl"),
            "main",
            &[(&input_buf, true), (&fsm_buf, false)],
            1,
        );

        let compact_buf =
            gpu.storage_buffer_empty("compact", (256 * std::mem::size_of::<u32>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_compact.wgsl"),
            "main",
            &[(&input_buf, true), (&fsm_buf, true), (&compact_buf, false)],
            1,
        );

        let depth_buf =
            gpu.storage_buffer_empty("depth", (256 * std::mem::size_of::<i32>()) as u64);

        gpu.dispatch(
            include_str!("shaders/scan_depth.wgsl"),
            "main",
            &[
                (&input_buf, true),
                (&compact_buf, true),
                (&depth_buf, false),
            ],
            1,
        );

        let parent_buf =
            gpu.storage_buffer_empty("parents", (256 * std::mem::size_of::<i32>()) as u64);

        gpu.dispatch(
            include_str!("shaders/parent_link.wgsl"),
            "main",
            &[
                (&input_buf, true),
                (&compact_buf, true),
                (&depth_buf, true),
                (&parent_buf, false),
            ],
            1,
        );

        let parents = gpu.read_buffer_as::<i32>(&parent_buf);
        let depths = gpu.read_buffer_as::<i32>(&depth_buf);
        let positions = gpu.read_buffer_as::<u32>(&compact_buf);

        assert_eq!(parents[0], -1);
        insta::assert_compact_debug_snapshot!(depths, @"[1, 1, 2, 2, 3, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]");
        insta::assert_compact_debug_snapshot!(positions, @"[0, 4, 5, 9, 10, 12, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]");
    }
}
