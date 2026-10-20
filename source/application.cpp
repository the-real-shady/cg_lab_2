#include "application.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <vulkan/vulkan.h>

#include "math.hpp"

namespace application {

namespace {

using graphics::internal::context;

struct Vertex {
	float position[3];
	float normal[3];
};

struct SceneUniforms {
	math::Mat4 view_projection;
	float camera_position[4];
	float ambient[4];
	float light_direction[4];
	float light_color[4];
	uint32_t light_counts[4];
};

struct PointLight {
	float position[4];
	float color[4];
};

struct SpotLight {
	float position[4];
	float direction[4];
	float color[4];
	float cone[4];
};

constexpr uint32_t max_point_lights = 128;
constexpr uint32_t max_spot_lights = 128;

struct ModelUniforms {
	math::Mat4 model;
	math::Mat4 normal_matrix;
	float albedo[4];
	float specular_color[4];
	float shininess;
};

constexpr math::Vec3 octahedron_positions[] = {
	{  1.0f,  0.0f,  0.0f },
	{ -1.0f,  0.0f,  0.0f },
	{  0.0f,  1.0f,  0.0f },
	{  0.0f, -1.0f,  0.0f },
	{  0.0f,  0.0f,  1.0f },
	{  0.0f,  0.0f, -1.0f },
};

constexpr uint32_t octahedron_faces[][3] = {
	{ 2, 4, 0 }, { 2, 1, 4 }, { 2, 5, 1 }, { 2, 0, 5 },
	{ 3, 0, 4 }, { 3, 4, 1 }, { 3, 1, 5 }, { 3, 5, 0 },
};

constexpr uint32_t face_count = sizeof(octahedron_faces) / sizeof(octahedron_faces[0]);
constexpr uint32_t vertex_count = face_count * 3;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

Buffer vertex_buffer;
Buffer scene_buffer;
Buffer point_light_buffer;
Buffer spot_light_buffer;
Buffer model_buffer;

VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_set;
VkDescriptorSet vk_model_set;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

struct Camera {
	math::Vec3 position = { 0.0f, 1.5f, 6.0f };
	float yaw = 0.0f;
	float pitch = -15.0f;
	float field_of_view = 60.0f;
	float move_speed = 4.0f;
};

struct DirectionalLight {
	float azimuth = 30.0f;
	float elevation = 40.0f;
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 1.0f;
};

struct Material {
	math::Vec3 albedo = { 0.8f, 0.45f, 0.2f };
	math::Vec3 specular_color = { 1.0f, 1.0f, 1.0f };
	float shininess = 64.0f;
};

struct Object {
	math::Vec3 position = { 0.0f, 0.0f, 0.0f };
	math::Vec3 rotation = { 20.0f, 30.0f, 0.0f };
	math::Vec3 scale = { 1.0f, 1.0f, 1.0f };
};

Camera camera;
DirectionalLight light;
Material material;
Object object;
struct PointLightSource {
	math::Vec3 position = { 0.0f, 2.0f, 2.0f };
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 10.0f;
};

std::vector<PointLightSource> point_lights = {
	{ .position = { -2.5f, 1.5f, 2.0f }, .color = { 1.0f, 0.3f, 0.2f }, .intensity = 12.0f },
	{ .position = { 2.5f, 1.0f, 2.0f }, .color = { 0.2f, 0.5f, 1.0f }, .intensity = 12.0f },
};

struct SpotLightSource {
	math::Vec3 position = { 0.0f, 4.0f, 0.0f };
	float azimuth = 0.0f;
	float elevation = -90.0f;
	math::Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 40.0f;
	float inner_angle = 15.0f;
	float outer_angle = 30.0f;
};

std::vector<SpotLightSource> spot_lights = {
	{ .position = { 0.0f, 4.0f, 1.0f }, .color = { 1.0f, 0.9f, 0.6f } },
};

math::Vec3 ambient_color = { 0.1f, 0.1f, 0.12f };

double previous_time;

math::Vec3 cameraForward() {
	const float yaw = math::radians(camera.yaw);
	const float pitch = math::radians(camera.pitch);

	return { std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch) };
}

math::Vec3 directionFromAngles(float azimuth_degrees, float elevation_degrees) {
	const float azimuth = math::radians(azimuth_degrees);
	const float elevation = math::radians(elevation_degrees);

	return { std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
	         std::cos(elevation) * std::cos(azimuth) };
}

math::Vec3 lightDirection() {
	return directionFromAngles(light.azimuth, light.elevation);
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result_buffer) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result_buffer.buffer, &result_buffer.allocation, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer of " << size << " bytes\n";
		return false;
	}

	result_buffer.mapped = result.pMappedData;

	return true;
}

bool loadShaderModule(const char* path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module for " << path << '\n';
		return false;
	}

	return true;
}

bool createPipelines() {
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE;

	if (!loadShaderModule("shaders/lit.vert.spv", vertex_shader) ||
	    !loadShaderModule("shaders/lit.frag.spv", fragment_shader)) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDescriptorSetLayout set_layouts[] = { vk_scene_set_layout, vk_model_set_layout };

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline, nullptr, &vk_pipeline);

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	const VkShaderStageFlags all_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = all_stages,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = all_stages,
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &model_layout, nullptr,
	                                &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = { vk_scene_set_layout, vk_model_set_layout };
	VkDescriptorSet sets[2];

	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	vk_scene_set = sets[0];
	vk_model_set = sets[1];

	const VkDescriptorBufferInfo buffer_infos[] = {
		{ .buffer = scene_buffer.buffer, .offset = 0, .range = sizeof(SceneUniforms) },
		{ .buffer = point_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
		{ .buffer = spot_light_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
		{ .buffer = model_buffer.buffer, .offset = 0, .range = sizeof(ModelUniforms) },
	};

	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &buffer_infos[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &buffer_infos[2],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_model_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[3],
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);

	return true;
}

}

bool initialize() {
	Vertex vertices[vertex_count] = {};

	for (uint32_t f = 0; f < face_count; ++f) {
		const math::Vec3 a = octahedron_positions[octahedron_faces[f][0]];
		const math::Vec3 b = octahedron_positions[octahedron_faces[f][1]];
		const math::Vec3 c = octahedron_positions[octahedron_faces[f][2]];

		math::Vec3 normal = math::normalize(math::cross(b - a, c - a));

		if (math::dot(normal, a + b + c) < 0.0f) {
			normal = normal * -1.0f;
		}

		const math::Vec3 face[3] = { a, b, c };
		for (uint32_t i = 0; i < 3; ++i) {
			vertices[f * 3 + i] = { { face[i].x, face[i].y, face[i].z },
			                        { normal.x, normal.y, normal.z } };
		}
	}

	if (!createBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) {
		return false;
	}
	std::memcpy(vertex_buffer.mapped, vertices, sizeof(vertices));

	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
	    !createBuffer(sizeof(PointLight) * max_point_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) ||
	    !createBuffer(sizeof(SpotLight) * max_spot_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer) ||
	    !createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer)) {
		return false;
	}

	return createDescriptors() && createPipelines();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	vmaDestroyBuffer(context.allocator, model_buffer.buffer, model_buffer.allocation);
	vmaDestroyBuffer(context.allocator, spot_light_buffer.buffer, spot_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, point_light_buffer.buffer, point_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, scene_buffer.buffer, scene_buffer.allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer.buffer, vertex_buffer.allocation);
}

void update(double time) {
	const float delta = previous_time > 0.0 ? float(time - previous_time) : 0.0f;
	previous_time = time;

	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw += io.MouseDelta.x * 0.2f;
		camera.pitch -= io.MouseDelta.y * 0.2f;
		camera.pitch = std::fmax(-89.0f, std::fmin(89.0f, camera.pitch));
	}

	if (!io.WantCaptureKeyboard) {
		const math::Vec3 forward = cameraForward();
		const math::Vec3 right = math::normalize(math::cross(forward, { 0.0f, 1.0f, 0.0f }));
		const float step = camera.move_speed * delta;

		if (ImGui::IsKeyDown(ImGuiKey_W)) camera.position = camera.position + forward * step;
		if (ImGui::IsKeyDown(ImGuiKey_S)) camera.position = camera.position - forward * step;
		if (ImGui::IsKeyDown(ImGuiKey_D)) camera.position = camera.position + right * step;
		if (ImGui::IsKeyDown(ImGuiKey_A)) camera.position = camera.position - right * step;
		if (ImGui::IsKeyDown(ImGuiKey_E)) camera.position.y += step;
		if (ImGui::IsKeyDown(ImGuiKey_Q)) camera.position.y -= step;
	}

	ImGui::Begin("Lab 2: camera and lighting");
	ImGui::Text("%.1f FPS", double(io.Framerate));

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextWrapped("RMB drag: look, WASD/QE: move");
		ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");
		ImGui::SliderFloat("Field of view", &camera.field_of_view, 20.0f, 110.0f, "%.0f deg");
		ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
	}

	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient", &ambient_color.x);
		ImGui::SeparatorText("Directional light");
		ImGui::SliderFloat("Azimuth", &light.azimuth, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat("Elevation", &light.elevation, -90.0f, 90.0f, "%.0f deg");
		ImGui::ColorEdit3("Light color", &light.color.x);
		ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 5.0f);

		ImGui::SeparatorText("Point lights");

		ImGui::BeginDisabled(point_lights.size() >= max_point_lights);
		if (ImGui::Button("Add point light")) {
			point_lights.push_back({});
		}
		ImGui::EndDisabled();

		int remove_index = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			PointLightSource& source = point_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNodeEx("Point light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %zu", i + 1)) {
				ImGui::DragFloat3("Position", &source.position.x, 0.05f);
				ImGui::ColorEdit3("Color", &source.color.x);
				ImGui::DragFloat("Intensity", &source.intensity, 0.1f, 0.0f, 1000.0f);

				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			point_lights.erase(point_lights.begin() + remove_index);
		}

		ImGui::SeparatorText("Spot lights");

		ImGui::BeginDisabled(spot_lights.size() >= max_spot_lights);
		if (ImGui::Button("Add spot light")) {
			spot_lights.push_back({});
		}
		ImGui::EndDisabled();

		remove_index = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			SpotLightSource& source = spot_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNodeEx("Spot light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %zu", i + 1)) {
				ImGui::DragFloat3("Position", &source.position.x, 0.05f);
				ImGui::SliderFloat("Azimuth", &source.azimuth, -180.0f, 180.0f, "%.0f deg");
				ImGui::SliderFloat("Elevation", &source.elevation, -90.0f, 90.0f, "%.0f deg");
				ImGui::ColorEdit3("Color", &source.color.x);
				ImGui::DragFloat("Intensity", &source.intensity, 0.5f, 0.0f, 5000.0f);

				if (ImGui::SliderFloat("Inner angle", &source.inner_angle, 0.0f, 89.0f, "%.0f deg")) {
					source.outer_angle = std::fmax(source.outer_angle, source.inner_angle);
				}
				if (ImGui::SliderFloat("Outer angle", &source.outer_angle, 1.0f, 90.0f, "%.0f deg")) {
					source.inner_angle = std::fmin(source.inner_angle, source.outer_angle);
				}

				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_index);
		}
	}

	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", &material.albedo.x);
		ImGui::ColorEdit3("Specular color", &material.specular_color.x);
		ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 512.0f, "%.0f",
		                   ImGuiSliderFlags_Logarithmic);
	}

	if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("Position##object", &object.position.x, -5.0f, 5.0f);
		ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat3("Scale", &object.scale.x, 0.1f, 3.0f);
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const math::Mat4 view = math::lookAt(camera.position, camera.position + cameraForward(),
	                                     { 0.0f, 1.0f, 0.0f });
	const math::Mat4 projection =
		math::perspective(math::radians(camera.field_of_view), aspect, 0.1f, 100.0f);

	const math::Vec3 light_dir = lightDirection();

	const SceneUniforms scene = {
		projection * view,
		{ camera.position.x, camera.position.y, camera.position.z, 1.0f },
		{ ambient_color.x, ambient_color.y, ambient_color.z, 1.0f },
		{ light_dir.x, light_dir.y, light_dir.z, 0.0f },
		{ light.color.x * light.intensity, light.color.y * light.intensity,
		  light.color.z * light.intensity, 1.0f },
		{ uint32_t(point_lights.size()), uint32_t(spot_lights.size()), 0, 0 },
	};

	const math::Mat4 rotation = math::rotate({ math::radians(object.rotation.x),
	                                           math::radians(object.rotation.y),
	                                           math::radians(object.rotation.z) });

	const math::Mat4 normal_matrix =
		rotation * math::scale({ 1.0f / object.scale.x, 1.0f / object.scale.y, 1.0f / object.scale.z });

	const ModelUniforms model = {
		math::translate(object.position) * rotation * math::scale(object.scale),
		normal_matrix,
		{ material.albedo.x, material.albedo.y, material.albedo.z, 1.0f },
		{ material.specular_color.x, material.specular_color.y, material.specular_color.z, 1.0f },
		material.shininess,
	};

	std::memcpy(scene_buffer.mapped, &scene, sizeof(scene));

	PointLight* gpu_point_lights = static_cast<PointLight*>(point_light_buffer.mapped);
	for (size_t i = 0; i < point_lights.size(); ++i) {
		const PointLightSource& source = point_lights[i];

		gpu_point_lights[i] = {
			{ source.position.x, source.position.y, source.position.z, 1.0f },
			{ source.color.x * source.intensity, source.color.y * source.intensity,
			  source.color.z * source.intensity, 1.0f },
		};
	}

	SpotLight* gpu_spot_lights = static_cast<SpotLight*>(spot_light_buffer.mapped);
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLightSource& source = spot_lights[i];
		const math::Vec3 direction = directionFromAngles(source.azimuth, source.elevation);

		gpu_spot_lights[i] = {
			{ source.position.x, source.position.y, source.position.z, 1.0f },
			{ direction.x, direction.y, direction.z, 0.0f },
			{ source.color.x * source.intensity, source.color.y * source.intensity,
			  source.color.z * source.intensity, 1.0f },
			{ std::cos(math::radians(source.inner_angle)),
			  std::cos(math::radians(source.outer_angle)), 0.0f, 0.0f },
		};
	}
	std::memcpy(model_buffer.mapped, &model, sizeof(model));

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.05f, 0.06f, 0.09f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	const VkDescriptorSet sets[] = { vk_scene_set, vk_model_set };
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 2, sets, 0, nullptr);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

}
