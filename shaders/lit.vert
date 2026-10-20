#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 frag_world_position;
layout(location = 1) out vec3 frag_normal;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uvec4 light_counts;
} scene;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular_color;
	float shininess;
} model;

void main() {
	const vec4 world_position = model.model * vec4(in_position, 1.0);

	gl_Position = scene.view_projection * world_position;

	frag_world_position = world_position.xyz;
	frag_normal = mat3(model.normal_matrix) * in_normal;
}
