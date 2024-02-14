#version 450
// This extension isn't supported or necessary for glslc.exe to compile this to SPIR-V.
// But it keeps my linter from complaining at me
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = vec4(inPosition, 0.0, 1.0);
	fragColor = inColor;
}