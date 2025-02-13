// A minimal particle-sandbox to demonstrate a reasonable use-case for a device-local buffer. We
// gain significant runtime performance by writing the initial vertex values to the GPU using a
// staging buffer and then copying the data to a device-local buffer to be accessed solely by the
// GPU through the compute shader and as a vertex array.

use std::{error::Error, sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{
        acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const PARTICLE_COUNT: usize = 100_000;

fn main() -> Result<(), impl Error> {
    // The usual Vulkan initialization. Largely the same as the triangle example until further
    // commentation is provided.

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    compute_pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vs: EntryPoint,
    fs: EntryPoint,
    pipeline: Arc<GraphicsPipeline>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    start_time: SystemTime,
    last_frame_time: SystemTime,
    cursor_pos: PhysicalPosition<f64>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
        let vertex_buffer = {
            // Initialize vertex data as an iterator.
            let vertices = (0..PARTICLE_COUNT).map(|i| {
                let f = i as f32 / (PARTICLE_COUNT / 10) as f32;
                MyVertex {
                    pos: [2. * f.fract() - 1., 0.2 * f.floor() - 1.],
                    vel: [0.; 2],
                }
            });

            // Create a CPU-accessible buffer initialized with the vertex data.
            let temporary_accessible_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    // Specify this buffer will be used as a transfer source.
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    // Specify this buffer will be used for uploading to the GPU.
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vertices,
            )
            .unwrap();

            // Create a buffer in device-local memory with enough space for `PARTICLE_COUNT`
            // number of `Vertex`.
            let device_local_buffer = Buffer::new_slice::<MyVertex>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    // Specify use as a storage buffer, vertex buffer, and transfer destination.
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST
                        | BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    // Specify this buffer will only be used by the device.
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                PARTICLE_COUNT as DeviceSize,
            )
            .unwrap();

            // Create one-time command to copy between the buffers.
            let mut cbb = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            cbb.copy_buffer(CopyBufferInfo::buffers(
                temporary_accessible_buffer,
                device_local_buffer.clone(),
            ))
            .unwrap();
            let cb = cbb.build().unwrap();

            // Execute copy and wait for copy to complete before proceeding.
            cb.execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None /* timeout */)
                .unwrap();

            device_local_buffer
        };

        // Create a compute-pipeline for applying the compute shader to vertices.
        let compute_pipeline = {
            let cs = cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        // Create a new descriptor set for binding vertices as a storage buffer.
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            // 0 is the index of the descriptor set.
            compute_pipeline.layout().set_layouts()[0].clone(),
            [
                // 0 is the binding of the data in this set. We bind the `Buffer` of vertices here.
                WriteDescriptorSet::buffer(0, vertex_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        App {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            vertex_buffer,
            compute_pipeline,
            descriptor_set,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        // For simplicity, we are going to assert that the window size is static.
                        .with_title("simple particles"),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    present_mode: PresentMode::Fifo,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let vs = vs::load(self.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(self.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let (framebuffers, pipeline) = window_size_dependent_setup(
            window_size,
            &images,
            &render_pass,
            &self.memory_allocator,
            &vs,
            &fs,
        );
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let start_time = SystemTime::now();

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            framebuffers,
            render_pass,
            pipeline,
            vs,
            fs,
            recreate_swapchain: false,
            previous_frame_end,
            start_time,
            last_frame_time: start_time,
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let window_size = rcx.window.inner_size();
                rcx.cursor_pos.x = 2.0 * (position.x / window_size.width as f64) - 1.0;
                rcx.cursor_pos.y = 2.0 * (position.y / window_size.height as f64) - 1.0;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;
                    (rcx.framebuffers, rcx.pipeline) = window_size_dependent_setup(
                        window_size,
                        &new_images,
                        &rcx.render_pass,
                        &self.memory_allocator,
                        &rcx.vs,
                        &rcx.fs,
                    );
                    rcx.recreate_swapchain = false;
                }

                // Update per-frame variables.
                let now = SystemTime::now();
                let time = now.duration_since(rcx.start_time).unwrap().as_secs_f32();
                let delta_time = now
                    .duration_since(rcx.last_frame_time)
                    .unwrap()
                    .as_secs_f32();
                rcx.last_frame_time = now;

                // Create push constants to be passed to compute shader.
                let push_constants = cs::PushConstants {
                    attractor: rcx.cursor_pos.into(),
                    attractor_strength: 1.2 * (2. * time).cos(),
                    delta_time,
                };

                // Acquire information on the next swapchain target.
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None, // timeout
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Push constants for compute shader.
                    .push_constants(self.compute_pipeline.layout().clone(), 0, push_constants)
                    .unwrap()
                    // Perform compute operation to update particle positions.
                    .bind_pipeline_compute(self.compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        self.descriptor_set.clone(),
                    )
                    .unwrap();
                unsafe { builder.dispatch([PARTICLE_COUNT as u32 / 128, 1, 1]) }.unwrap();

                // Use render-pass to draw particles to swapchain.
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0., 0., 0., 1.].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap();
                unsafe { builder.draw(PARTICLE_COUNT as u32, 1, 0, 0) }.unwrap();

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();
                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                rcx.previous_frame_end = match future.map_err(Validated::unwrap) {
                    // Success, store result into vector.
                    Ok(future) => Some(future.boxed()),
                    // Unknown failure.
                    Err(e) => panic!("failed to flush future: {e}"),
                };
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

fn window_size_dependent_setup(
    window_size: PhysicalSize<u32>,
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vs: &EntryPoint,
    fs: &EntryPoint,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
    let device = memory_allocator.device();

    let framebuffers = images
        .iter()
        .map(|img| {
            let view = ImageView::new_default(img.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    // Create a basic graphics pipeline for rendering particles.
    let pipeline = {
        let vertex_input_state = MyVertex::per_vertex().definition(vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                // Vertices will be rendered as a list of points.
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::PointList,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: window_size.into(),
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };
    (framebuffers, pipeline)
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    vel: [f32; 2],
}

// The vertex shader determines color and is run once per particle. The vertices will be
// updated by the compute shader each frame.
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 vel;

            layout(location = 0) out vec4 outColor;

            // Keep this value in sync with the `maxSpeed` const in the compute shader.
            const float maxSpeed = 10.0;

            void main() {
                gl_Position = vec4(pos, 0.0, 1.0);
                gl_PointSize = 1.0;

                // Mix colors based on position and velocity.
                outColor = mix(
                    0.2 * vec4(pos, abs(vel.x) + abs(vel.y), 1.0),
                    vec4(1.0, 0.5, 0.8, 1.0),
                    sqrt(length(vel) / maxSpeed)
                );
            }
        ",
    }
}

// The fragment shader will only need to apply the color forwarded by the vertex shader,
// because the color of a particle should be identical over all pixels.
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec4 outColor;

            layout(location = 0) out vec4 fragColor;

            void main() {
                fragColor = outColor;
            }
        ",
    }
}

// Compute shader for updating the position and velocity of each particle every frame.
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            struct VertexData {
                vec2 pos;
                vec2 vel;
            };

            // Storage buffer binding, which we optimize by using a DeviceLocalBuffer.
            layout (binding = 0) buffer VertexBuffer {
                VertexData vertices[];
            };

            // Allow push constants to define a parameters of compute.
            layout (push_constant) uniform PushConstants {
                vec2 attractor;
                float attractor_strength;
                float delta_time;
            } push;

            // Keep this value in sync with the `maxSpeed` const in the vertex shader.
            const float maxSpeed = 10.0;

            const float minLength = 0.02;
            const float friction = -2.0;

            void main() {
                const uint index = gl_GlobalInvocationID.x;

                vec2 vel = vertices[index].vel;

                // Update particle position according to velocity.
                vec2 pos = vertices[index].pos + push.delta_time * vel;

                // Bounce particle off screen-border.
                if (abs(pos.x) > 1.0) {
                    vel.x = sign(pos.x) * (-0.95 * abs(vel.x) - 0.0001);
                    if (abs(pos.x) >= 1.05) {
                        pos.x = sign(pos.x);
                    }
                }
                if (abs(pos.y) > 1.0) {
                    vel.y = sign(pos.y) * (-0.95 * abs(vel.y) - 0.0001);
                    if (abs(pos.y) >= 1.05) {
                        pos.y = sign(pos.y);
                    }
                }

                // Simple inverse-square force.
                vec2 t = push.attractor - pos;
                float r = max(length(t), minLength);
                vec2 force = push.attractor_strength * (t / r) / (r * r);

                // Update velocity, enforcing a maximum speed.
                vel += push.delta_time * force;
                if (length(vel) > maxSpeed) {
                    vel = maxSpeed*normalize(vel);
                }

                // Set new values back into buffer.
                vertices[index].pos = pos;
                vertices[index].vel = vel * exp(friction * push.delta_time);
            }
        ",
    }
}
