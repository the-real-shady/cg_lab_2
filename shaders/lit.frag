#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uvec4 light_counts;
} scene;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 cone;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular_color;
	float shininess;
} model;

vec3 blinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction, vec3 radiance) {
	const vec3 half_vector = normalize(light_direction + view_direction);

	const float n_dot_l = max(dot(normal, light_direction), 0.0);
	const float n_dot_h = max(dot(normal, half_vector), 0.0);

	const vec3 diffuse = model.albedo.rgb * n_dot_l;
	const vec3 specular = n_dot_l > 0.0
		? model.specular_color.rgb * pow(n_dot_h, model.shininess)
		: vec3(0.0);

	return (diffuse + specular) * radiance;
}

void main() {
	const vec3 normal = normalize(frag_normal);
	const vec3 view_direction = normalize(scene.camera_position.xyz - frag_world_position);

	vec3 color = scene.ambient.rgb * model.albedo.rgb;

	color += blinnPhong(normal, view_direction, normalize(scene.light_direction.xyz),
	                    scene.light_color.rgb);

	for (uint i = 0; i < scene.light_counts.x; ++i) {
		const vec3 to_light = point_lights[i].position.xyz - frag_world_position;
		const float distance_squared = max(dot(to_light, to_light), 0.0001);
		const float attenuation = 1.0 / distance_squared;

		color += blinnPhong(normal, view_direction, to_light * inversesqrt(distance_squared),
		                    point_lights[i].color.rgb * attenuation);
	}

	for (uint i = 0; i < scene.light_counts.y; ++i) {
		const vec3 to_light = spot_lights[i].position.xyz - frag_world_position;
		const float distance_squared = max(dot(to_light, to_light), 0.0001);
		const vec3 light_direction = to_light * inversesqrt(distance_squared);

		const float cos_inner = spot_lights[i].cone.x;
		const float cos_outer = spot_lights[i].cone.y;
		const float cos_angle = dot(-light_direction, normalize(spot_lights[i].direction.xyz));
		const float cone = smoothstep(0.0, 1.0,
			clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0));

		const float attenuation = cone / distance_squared;

		color += blinnPhong(normal, view_direction, light_direction,
		                    spot_lights[i].color.rgb * attenuation);
	}

	out_color = vec4(color, 1.0);
}
