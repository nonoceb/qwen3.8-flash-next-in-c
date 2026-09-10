# Makefile for Vulkan compute shaders
# Compiles GLSL compute shaders to SPIR-V

SHADER_DIR = ../shaders
SPV_DIR = ../shaders/spv

SHADERS = \
	$(SHADER_DIR)/gemv_f32.comp \
	$(SHADER_DIR)/gemv_f32_optimized.comp \
	$(SHADER_DIR)/gemv_iq4nl.comp \
	$(SHADER_DIR)/gemv_q8_0.comp \
	$(SHADER_DIR)/gemm_f32_tiled.comp

SPV_SHADERS = $(patsubst $(SHADER_DIR)/%.comp,$(SPV_DIR)/%.spv,$(SHADERS))

.PHONY: all clean

all: $(SPV_DIR) $(SPV_SHADERS)
	@echo "All shaders compiled to SPIR-V"

$(SPV_DIR):
	mkdir -p $(SPV_DIR)

$(SPV_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@echo "Compiling $< -> $@"
	glslc $< -o $@ || glslangValidator -V $< -o $@

clean:
	rm -rf $(SPV_DIR)
