# Pinned upstream llama.cpp (b1b770d9) with Metal and CPU, built without cmake,
# for the native engine's golden capture and baselines (docs/NATIVE_PARITY.md,
# docs/NATIVE_BENCHMARKS.md). It builds the same sources the vendored
# Sources/LlamaCpp target compiles (Sources/LlamaCpp/vendoring/vendor.sh --check
# proves the files are byte-identical) with the same defines, plus the tools
# below. Nothing is downloaded; the checkout must be at the pin and clean.
#
#   make -f Scripts/native/llama/Makefile LL=<llama.cpp checkout> OUT=<build dir> -j8 [targets]
#
# Targets: mfe-golden (default), llama-bench, all-tools.
# No metallib is embedded: at run time set
#   GGML_METAL_PATH_RESOURCES=$(LL)/ggml/src/ggml-metal
# and ggml compiles kernels/*.metal from source, as the vendored target does.

# LL (the upstream checkout at the pin) and OUT (the build directory, outside
# the repository) are required.
ifeq ($(strip $(LL)),)
  $(error set LL=<llama.cpp checkout at b1b770d9>)
endif
ifeq ($(strip $(OUT)),)
  $(error set OUT=<build directory>)
endif
HERE    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
NATIVE  := $(abspath $(HERE)/..)
PIN     := b1b770d90d94cad9e9ea5b993bdbc383a522fb28

CC   := clang
CXX  := clang++
OPT  := -O3 -DNDEBUG -mcpu=native -fno-fast-math
DEFS := -D_DARWIN_C_SOURCE -D_XOPEN_SOURCE=600 -DGGML_SCHED_MAX_COPIES=4 -DGGML_USE_CPU -DGGML_USE_METAL
INC  := -I$(OUT)/gen -I$(LL)/ggml/include -I$(LL)/ggml/src -I$(LL)/ggml/src/ggml-cpu -I$(LL)/ggml/src/ggml-metal \
        -I$(LL)/include -I$(LL)/common -I$(LL)/src -I$(LL)/vendor -I$(LL)/vendor/cpp-httplib -I$(LL)/tools/llama-bench
CPU_DEFS := -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_CPU_REPACK
HTTPLIB_DEFS := -DCPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576 -DCPPHTTPLIB_LISTEN_BACKLOG=512

CFLAGS    := -std=c11 $(OPT) $(DEFS) $(INC) -w
CXXFLAGS  := -std=c++17 $(OPT) $(DEFS) $(INC) -w
OBJCFLAGS := $(OPT) $(DEFS) $(INC) -fno-objc-arc -w
LDFLAGS   := -framework Accelerate -framework Foundation -framework Metal -framework MetalKit

BASE_C   := ggml.c ggml-alloc.c ggml-quants.c
BASE_CXX := ggml.cpp ggml-backend.cpp ggml-backend-meta.cpp ggml-opt.cpp ggml-threading.cpp gguf.cpp \
            ggml-backend-reg.cpp ggml-backend-dl.cpp
BASE_OBJ := $(addprefix $(OUT)/ggml/,$(BASE_C:.c=.o) $(BASE_CXX:.cpp=.o))

CPU_C_OBJ := $(OUT)/cpu/ggml-cpu-c.o $(OUT)/cpu/quants.o $(OUT)/cpu/arm-quants.o
CPU_CXX_SRC := $(addprefix $(LL)/ggml/src/ggml-cpu/,ggml-cpu.cpp repack.cpp hbm.cpp traits.cpp amx/amx.cpp amx/mmq.cpp \
               tiled/tiled.cpp tiled/tiled-kernel.cpp binary-ops.cpp unary-ops.cpp vec.cpp ops.cpp arch/arm/repack.cpp \
               llamafile/sgemm.cpp)
CPU_OBJ := $(CPU_C_OBJ) $(foreach f,$(CPU_CXX_SRC),$(OUT)/cpu/$(subst /,_,$(patsubst $(LL)/ggml/src/ggml-cpu/%,%,$(f:.cpp=.o))))

METAL_CXX := ggml-metal.cpp ggml-metal-device.cpp ggml-metal-common.cpp ggml-metal-fusion.cpp ggml-metal-ops.cpp ggml-metal-tuning.cpp
METAL_M   := ggml-metal-device.m ggml-metal-context.m
METAL_OBJ := $(addprefix $(OUT)/metal/,$(METAL_CXX:.cpp=.o) $(METAL_M:.m=-m.o))

GGML_OBJ := $(BASE_OBJ) $(CPU_OBJ) $(METAL_OBJ)

LLAMA_SRC := $(wildcard $(LL)/src/*.cpp) $(wildcard $(LL)/src/models/*.cpp)
LLAMA_OBJ := $(patsubst $(LL)/src/%.cpp,$(OUT)/llama/%.o,$(LLAMA_SRC))

COMMON_CXX := arg.cpp chat-auto-parser-generator.cpp chat-auto-parser-helpers.cpp chat-diff-analyzer.cpp \
              chat-peg-parser.cpp chat-tts-norm.cpp chat.cpp common.cpp console.cpp debug.cpp download.cpp fit.cpp \
              file-lock.cpp hf-cache.cpp imatrix-loader.cpp json-schema-to-grammar.cpp json-schema.cpp json.cpp \
              llguidance.cpp log.cpp ngram-cache.cpp ngram-map.cpp ngram-mod.cpp peg-parser.cpp preset.cpp \
              reasoning-budget.cpp sampling.cpp speculative.cpp subproc.cpp trie.cpp unicode.cpp \
              jinja/lexer.cpp jinja/parser.cpp jinja/runtime.cpp jinja/value.cpp jinja/string.cpp jinja/caps.cpp \
              $(patsubst $(LL)/common/%,%,$(wildcard $(LL)/common/parsers/*.cpp))
COMMON_OBJ := $(addprefix $(OUT)/common/,$(subst /,_,$(COMMON_CXX:.cpp=.o))) $(OUT)/common/build-info.o $(OUT)/common/httplib.o

all: $(OUT)/mfe-golden $(OUT)/BUILD_INFO.txt
all-tools: $(OUT)/mfe-golden $(OUT)/llama-bench $(OUT)/BUILD_INFO.txt
mfe-golden: $(OUT)/mfe-golden
llama-bench: $(OUT)/llama-bench

pin-check:
	@test "$$(git -C $(LL) rev-parse HEAD)" = "$(PIN)" || { echo "llama.cpp checkout $(LL) is not at $(PIN)"; exit 1; }
	@test -z "$$(git -C $(LL) status --porcelain)" || { echo "llama.cpp checkout $(LL) is dirty"; exit 1; }

GEN_H := $(OUT)/gen/ggml-version.h
$(GEN_H): | pin-check
	@mkdir -p $(dir $@)
	printf '#pragma once\n#define GGML_VERSION "1.0.0"\n#define GGML_COMMIT  "b1b770d9"\n' > $@
	printf '#pragma once\n#define LLAMA_VERSION "1.0.0"\n#define LLAMA_COMMIT  "b1b770d9"\n' > $(dir $@)llama-version.h
$(OUT)/gen/build-info.cpp: $(LL)/common/build-info.cpp.in
	@mkdir -p $(dir $@)
	sed -e 's/@LLAMA_BUILD_NUMBER@/0/' -e 's/@LLAMA_BUILD_COMMIT@/b1b770d9/' \
	    -e 's/@BUILD_COMPILER@/clang/' -e 's/@BUILD_TARGET@/arm64-apple-darwin-metal/' $< > $@

ALL_OBJ = $(GGML_OBJ) $(LLAMA_OBJ) $(COMMON_OBJ)
$(ALL_OBJ): | $(GEN_H)

$(OUT)/ggml/%.o: $(LL)/ggml/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/ggml/%.o: $(LL)/ggml/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/cpu/ggml-cpu-c.o: $(LL)/ggml/src/ggml-cpu/ggml-cpu.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPU_DEFS) -c $< -o $@
$(OUT)/cpu/quants.o: $(LL)/ggml/src/ggml-cpu/quants.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPU_DEFS) -c $< -o $@
$(OUT)/cpu/arm-quants.o: $(LL)/ggml/src/ggml-cpu/arch/arm/quants.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPU_DEFS) -c $< -o $@
define cpu_cxx_rule
$(OUT)/cpu/$(subst /,_,$(patsubst $(LL)/ggml/src/ggml-cpu/%,%,$(1:.cpp=.o))): $(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) $(CPU_DEFS) -c $$< -o $$@
endef
$(foreach f,$(CPU_CXX_SRC),$(eval $(call cpu_cxx_rule,$(f))))

$(OUT)/metal/%.o: $(LL)/ggml/src/ggml-metal/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/metal/%-m.o: $(LL)/ggml/src/ggml-metal/%.m
	@mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(OUT)/llama/%.o: $(LL)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

define common_rule
$(OUT)/common/$(subst /,_,$(1:.cpp=.o)): $(LL)/common/$(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) -c $$< -o $$@
endef
$(foreach f,$(COMMON_CXX),$(eval $(call common_rule,$(f))))
$(OUT)/common/build-info.o: $(OUT)/gen/build-info.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/common/httplib.o: $(LL)/vendor/cpp-httplib/httplib.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(HTTPLIB_DEFS) -c $< -o $@

# --- tools
$(OUT)/tools/mfe-golden.o: $(NATIVE)/golden/mfe_golden.cpp | $(GEN_H)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/mfe-golden: $(GGML_OBJ) $(LLAMA_OBJ) $(COMMON_OBJ) $(OUT)/tools/mfe-golden.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OUT)/bench/%.o: $(LL)/tools/llama-bench/%.cpp | $(GEN_H)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/llama-bench: $(GGML_OBJ) $(LLAMA_OBJ) $(COMMON_OBJ) $(OUT)/bench/llama-bench.o $(OUT)/bench/main.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OUT)/BUILD_INFO.txt: $(OUT)/mfe-golden
	{ echo "llama.cpp $$(git -C $(LL) rev-parse HEAD) dirty=$$(git -C $(LL) status --porcelain | wc -l | tr -d ' ')"; \
	  echo "variant metal+cpu (Scripts/native/llama/Makefile)"; clang --version | head -1; \
	  echo "CXXFLAGS $(CXXFLAGS) $(CPU_DEFS)"; \
	  for b in $(OUT)/mfe-golden $(OUT)/llama-bench; do [ -f $$b ] && shasum -a 256 $$b; done; true; } > $@

.PHONY: all all-tools mfe-golden llama-bench pin-check
