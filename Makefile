NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=native
PYTHON ?= python3
PNPM ?= pnpm
CUDA_DIR := operators/cuda/vector_add
BUILD_DIR ?= build/cuda/vector_add
EXAMPLES := vector_add test_boundaries test_random benchmark_events benchmark_roundtrip
BINARIES := $(addprefix $(BUILD_DIR)/,$(EXAMPLES))

.PHONY: help check cuda test-cuda benchmark-cuda docs-dev docs-build docs-check

help:
	@echo "make check          - Check repository structure, local links and example consistency"
	@echo "make cuda           - Compile all first-week CUDA examples (requires nvcc)"
	@echo "make test-cuda      - Run baseline, boundary and random tests (requires NVIDIA GPU)"
	@echo "make benchmark-cuda - Run event and roundtrip timing examples"
	@echo "make docs-dev       - Start the reference documentation site"
	@echo "make docs-check     - Build the site and check links, anchors and source coverage"

check:
	$(PYTHON) scripts/check_repository.py

cuda: $(BINARIES)

$(BUILD_DIR)/%: $(CUDA_DIR)/%.cu
	@mkdir -p "$(BUILD_DIR)"
	$(NVCC) $(NVCCFLAGS) "$<" -o "$@"

test-cuda: $(BUILD_DIR)/vector_add $(BUILD_DIR)/test_boundaries $(BUILD_DIR)/test_random
	"$(BUILD_DIR)/vector_add"
	"$(BUILD_DIR)/test_boundaries"
	"$(BUILD_DIR)/test_random"

benchmark-cuda: $(BUILD_DIR)/benchmark_events $(BUILD_DIR)/benchmark_roundtrip
	"$(BUILD_DIR)/benchmark_events"
	"$(BUILD_DIR)/benchmark_roundtrip"

docs-dev:
	$(PNPM) --dir cuda-docs dev

docs-build:
	$(PNPM) --dir cuda-docs build

docs-check: docs-build
	$(PNPM) --dir cuda-docs check
