# Stock upstream llama.cpp (a release tag that contains #32704 and #32799),
# Metal + CPU, built without cmake for the Inkling program's base reference
# (scratch/inkling-program spec, section 3). It mirrors what upstream's
# CMakeLists.txt builds by default on Apple arm64: ggml-base, the CPU backend
# (GGML_NATIVE, LLAMAFILE, CPU_REPACK, ACCELERATE), the BLAS backend
# (Accelerate), the Metal backend with its kernels embedded
# (GGML_METAL_EMBED_LIBRARY), libllama (src/*.cpp, src/models/*.cpp),
# llama-common (with LLAMA_SUBPROCESS), mtmd and the server libraries. Not
# built: OpenSSL (no HTTPS downloads), llguidance, the web UI (an empty asset
# table, as `LLAMA_BUILD_UI=OFF` with no prebuilt UI gives), the tensor-API
# metallib (needs an M5-class GPU). Nothing is downloaded.
#
#   make -f Scripts/parity/inkling2/llama/Makefile LL=<checkout> OUT=<build dir> TAG=<bNNNNN> -j4 [targets]
#
# Targets: all (the tools below and BUILD_INFO.txt), llama-cli, llama-server,
# llama-bench, llama-completion, llama-eval-callback, llama-tokenize,
# llama-gguf, ink-golden (Scripts/parity/inkling2/ink_golden.cpp).
# The checkout must be clean and at the named tag (pin-check).

ifeq ($(strip $(LL)),)
  $(error set LL=<llama.cpp checkout>)
endif
ifeq ($(strip $(OUT)),)
  $(error set OUT=<build directory>)
endif
ifeq ($(strip $(TAG)),)
  $(error set TAG=<release tag, e.g. b12641>)
endif
HERE    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PARITY  := $(abspath $(HERE)/..)
COMMIT  := $(shell git -C $(LL) rev-parse HEAD)
SHORT   := $(shell git -C $(LL) rev-parse --short=8 HEAD)
BUILDNO := $(patsubst b%,%,$(TAG))
# parentheses for the sed ranges inside $(shell ...), which make would count
LP := (
RP := )

CC   := clang
CXX  := clang++
OPT  := -O3 -DNDEBUG -mcpu=native -fno-fast-math
DEFS := -D_DARWIN_C_SOURCE -D_XOPEN_SOURCE=600 -DGGML_SCHED_MAX_COPIES=4 \
        -DGGML_USE_CPU -DGGML_USE_METAL -DGGML_USE_BLAS -DLLAMA_SUBPROCESS
INC  := -I$(OUT)/gen -I$(LL)/ggml/include -I$(LL)/include -I$(LL)/vendor -I$(LL)/vendor/cpp-httplib
# ggml's own source dirs are private to ggml (ggml-cpu has a common.h too)
GGML_INC := -I$(LL)/ggml/src -I$(LL)/ggml/src/ggml-cpu -I$(LL)/ggml/src/ggml-metal
# As in upstream's CMake, src/ is private to libllama and common/ is what its
# users see (both have a unicode.h): set per object group below.
SCOPE_INC = -I$(LL)/common
CPU_DEFS   := -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_CPU_REPACK
BLAS_DEFS  := -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_BLAS_USE_ACCELERATE
METAL_DEFS := -DGGML_METAL_EMBED_LIBRARY
HTTPLIB_DEFS := -DCPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576 -DCPPHTTPLIB_LISTEN_BACKLOG=512 \
                -DCPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768 -DCPPHTTPLIB_TCP_NODELAY=1
MTMD_DEFS  := -D_USE_MATH_DEFINES -DMTMD_VIDEO
SERVER_INC := -I$(LL)/tools/server -I$(LL)/tools/mtmd -I$(LL) -I$(OUT)/gen/ui

CFLAGS    = -std=c11 $(OPT) $(DEFS) $(INC) $(SCOPE_INC) -w
CXXFLAGS  = -std=c++17 $(OPT) $(DEFS) $(INC) $(SCOPE_INC) -w
OBJCFLAGS = $(OPT) $(DEFS) $(INC) $(SCOPE_INC) -fno-objc-arc -w
LDFLAGS   := -framework Accelerate -framework Foundation -framework Metal -framework MetalKit

# ---------------------------------------------------------------- ggml
BASE_C   := ggml.c ggml-alloc.c ggml-quants.c
BASE_CXX := ggml-backend.cpp ggml-backend-meta.cpp ggml-opt.cpp ggml-threading.cpp gguf.cpp \
            ggml-backend-reg.cpp ggml-backend-dl.cpp
# ggml.c and ggml.cpp would both map to ggml.o; ggml.cpp (the uncaught-exception
# backtrace handler) gets its own object name
BASE_OBJ := $(addprefix $(OUT)/ggml/,$(BASE_C:.c=.o) $(BASE_CXX:.cpp=.o)) $(OUT)/ggml/ggml-cpp.o

CPU_C_OBJ := $(OUT)/cpu/ggml-cpu-c.o $(OUT)/cpu/quants.o $(OUT)/cpu/arm-quants.o
CPU_CXX_SRC := $(addprefix $(LL)/ggml/src/ggml-cpu/,ggml-cpu.cpp repack.cpp hbm.cpp traits.cpp amx/amx.cpp amx/mmq.cpp \
               tiled/tiled.cpp tiled/tiled-kernel.cpp binary-ops.cpp unary-ops.cpp vec.cpp ops.cpp arch/arm/repack.cpp \
               llamafile/sgemm.cpp)
CPU_OBJ := $(CPU_C_OBJ) $(foreach f,$(CPU_CXX_SRC),$(OUT)/cpu/$(subst /,_,$(patsubst $(LL)/ggml/src/ggml-cpu/%,%,$(f:.cpp=.o))))

BLAS_OBJ := $(OUT)/blas/ggml-blas.o

METAL_CXX := ggml-metal.cpp ggml-metal-device.cpp ggml-metal-common.cpp ggml-metal-fusion.cpp ggml-metal-ops.cpp ggml-metal-tuning.cpp
METAL_M   := ggml-metal-device.m ggml-metal-context.m
# the kernel list is upstream's own (METALLIB_KERNEL_SOURCES)
METAL_KINDS := $(shell sed -n '/^set$(LP)METALLIB_KERNEL_SOURCES/,/^$(RP)/p' $(LL)/ggml/src/ggml-metal/CMakeLists.txt | sed -n 's|.*kernels/\([a-z0-9_]*\)\.metal.*|\1|p')
METAL_EMBED_OBJ := $(foreach k,$(METAL_KINDS),$(OUT)/metal/embed/ggml-metal-embed-$(k).o)
METAL_OBJ := $(addprefix $(OUT)/metal/,$(METAL_CXX:.cpp=.o) $(METAL_M:.m=-m.o)) $(METAL_EMBED_OBJ)

GGML_OBJ := $(BASE_OBJ) $(CPU_OBJ) $(BLAS_OBJ) $(METAL_OBJ)

# ---------------------------------------------------------------- llama, common, vendor
LLAMA_SRC := $(wildcard $(LL)/src/*.cpp) $(wildcard $(LL)/src/models/*.cpp)
LLAMA_OBJ := $(patsubst $(LL)/src/%.cpp,$(OUT)/llama/%.o,$(LLAMA_SRC))

COMMON_CXX := arg.cpp chat-auto-parser-generator.cpp chat-auto-parser-helpers.cpp chat-diff-analyzer.cpp \
              chat-peg-parser.cpp chat-tts-norm.cpp chat.cpp common.cpp console.cpp debug.cpp download.cpp fit.cpp \
              file-lock.cpp hf-cache.cpp imatrix-loader.cpp json-schema-to-grammar.cpp json-schema.cpp json.cpp \
              llguidance.cpp log.cpp ngram-cache.cpp ngram-map.cpp ngram-mod.cpp peg-parser.cpp preset.cpp \
              reasoning-budget.cpp sampling.cpp speculative.cpp subproc.cpp trie.cpp unicode.cpp \
              jinja/lexer.cpp jinja/parser.cpp jinja/runtime.cpp jinja/value.cpp jinja/string.cpp jinja/caps.cpp \
              $(shell sed -n 's|.*CMAKE_CURRENT_LIST_DIR}/\([a-z0-9_.-]*\.cpp\).*|parsers/\1|p' $(LL)/common/parsers/sources.cmake)
COMMON_OBJ := $(addprefix $(OUT)/common/,$(subst /,_,$(COMMON_CXX:.cpp=.o))) $(OUT)/common/build-info.o
HTTPLIB_OBJ := $(OUT)/vendor/httplib.o
HASH_OBJ := $(OUT)/vendor/hash.o $(OUT)/vendor/xxhash.o $(OUT)/vendor/sha1.o $(OUT)/vendor/sha256.o

# ---------------------------------------------------------------- mtmd, server, cli
MTMD_CXX := $(shell sed -n '/^set$(LP)MTMD_CORE_SOURCES/,/^$(RP)/p' $(LL)/tools/mtmd/CMakeLists.txt | sed -n 's|^ *\([a-z0-9_./-]*\.cpp\).*|\1|p') \
            mtmd-helper-video-ffmpeg.cpp \
            $(shell sed -n '/^add_library$(LP)mtmd/,/^ *$(RP)/p' $(LL)/tools/mtmd/CMakeLists.txt | sed -n 's|^ *\(models/[a-z0-9_./-]*\.cpp\).*|\1|p')
MTMD_OBJ := $(addprefix $(OUT)/mtmd/,$(subst /,_,$(MTMD_CXX:.cpp=.o)))

SERVER_CTX_CXX := $(shell sed -n '/^set$(LP)TARGET server-context$(RP)/,/^$(RP)/p' $(LL)/tools/server/CMakeLists.txt | sed -n 's|^ *\([a-z0-9_.-]*\.cpp\).*|\1|p')
SERVER_IMPL_CXX := $(shell sed -n '/^set$(LP)TARGET llama-server-impl$(RP)/,/^$(RP)/p' $(LL)/tools/server/CMakeLists.txt | sed -n 's|^ *\([a-z0-9_.-]*\.cpp\).*|\1|p')
SERVER_OBJ := $(addprefix $(OUT)/server/,$(SERVER_CTX_CXX:.cpp=.o) $(SERVER_IMPL_CXX:.cpp=.o)) $(OUT)/server/ui.o
CLI_OBJ := $(addprefix $(OUT)/cli/,cli.o cli-client.o cli-context.o)

CORE_OBJ := $(GGML_OBJ) $(LLAMA_OBJ) $(COMMON_OBJ) $(HTTPLIB_OBJ)

TOOLS := $(OUT)/llama-cli $(OUT)/llama-server $(OUT)/llama-bench $(OUT)/llama-completion \
         $(OUT)/llama-eval-callback $(OUT)/llama-tokenize $(OUT)/llama-gguf

all: $(TOOLS) $(OUT)/BUILD_INFO.txt
llama-cli: $(OUT)/llama-cli
llama-server: $(OUT)/llama-server
llama-bench: $(OUT)/llama-bench
llama-completion: $(OUT)/llama-completion
llama-eval-callback: $(OUT)/llama-eval-callback
llama-tokenize: $(OUT)/llama-tokenize
llama-gguf: $(OUT)/llama-gguf
ink-golden: $(OUT)/ink-golden

pin-check:
	@test "$$(git -C $(LL) describe --tags --exact-match HEAD 2>/dev/null)" = "$(TAG)" || \
	  test "$$(git -C $(LL) rev-parse refs/tags/$(TAG) 2>/dev/null)" = "$(COMMIT)" || \
	  { echo "llama.cpp checkout $(LL) is not at tag $(TAG)"; exit 1; }
	@test -z "$$(git -C $(LL) status --porcelain)" || { echo "llama.cpp checkout $(LL) is dirty"; exit 1; }

GEN_H := $(OUT)/gen/ggml-version.h
$(GEN_H): | pin-check
	@mkdir -p $(dir $@)
	sed -e 's/@GGML_VERSION@/1.0.0/' -e 's/@GGML_BUILD_COMMIT@/$(SHORT)/' -e 's/@GGML_VERSION_MAJOR@/1/' \
	    -e 's/@GGML_VERSION_MINOR@/0/' -e 's/@GGML_VERSION_PATCH@/0/' $(LL)/ggml/src/ggml-version.h.in > $@
	sed -e 's/@LLAMA_VERSION@/1.0.0/' -e 's/@LLAMA_BUILD_COMMIT@/$(SHORT)/' $(LL)/src/llama-version.h.in > $(dir $@)llama-version.h
$(OUT)/gen/build-info.cpp: $(LL)/common/build-info.cpp.in | $(GEN_H)
	@mkdir -p $(dir $@)
	sed -e 's/@LLAMA_BUILD_NUMBER@/$(BUILDNO)/' -e 's/@LLAMA_BUILD_COMMIT@/$(SHORT)/' \
	    -e 's/@BUILD_COMPILER@/clang/' -e 's/@BUILD_TARGET@/arm64-apple-darwin-metal/' $< > $@
# no web UI: the empty asset table of scripts/ui-assets.cmake
$(OUT)/gen/ui/ui.h: $(LL)/ui/apps/chat/ui.h.in | $(GEN_H)
	@mkdir -p $(dir $@)
	sed -e 's|^#cmakedefine LLAMA_UI_HAS_ASSETS 1|/* #undef LLAMA_UI_HAS_ASSETS */|' -e 's/@N_ASSETS@/0/g' $< > $@
$(OUT)/gen/ui/ui.cpp: $(LL)/ui/apps/chat/ui.cpp.in | $(GEN_H)
	@mkdir -p $(dir $@)
	sed -e 's/@ASSET_ARRAYS@//' -e 's/@ASSET_TABLE@//' -e 's/@N_ASSETS@/0/g' -e 's/@USE_GZIP@/false/' $< > $@

ALL_OBJ = $(CORE_OBJ) $(HASH_OBJ) $(MTMD_OBJ) $(SERVER_OBJ) $(CLI_OBJ)
$(ALL_OBJ): | $(GEN_H)
$(GGML_OBJ) $(OUT)/tools/gguf.o: SCOPE_INC := $(GGML_INC)
$(LLAMA_OBJ): SCOPE_INC := -I$(LL)/src

# ggml
$(OUT)/ggml/%.o: $(LL)/ggml/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/ggml/%.o: $(LL)/ggml/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/ggml/ggml-cpp.o: $(LL)/ggml/src/ggml.cpp
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
	$(CXX) $$(CXXFLAGS) $(CPU_DEFS) -c $$< -o $$@
endef
$(foreach f,$(CPU_CXX_SRC),$(eval $(call cpu_cxx_rule,$(f))))

$(OUT)/blas/ggml-blas.o: $(LL)/ggml/src/ggml-blas/ggml-blas.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(BLAS_DEFS) -c $< -o $@

$(OUT)/metal/%.o: $(LL)/ggml/src/ggml-metal/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(METAL_DEFS) -c $< -o $@
$(OUT)/metal/%-m.o: $(LL)/ggml/src/ggml-metal/%.m
	@mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) $(METAL_DEFS) -c $< -o $@
$(OUT)/metal/embed/ggml-metal-embed-%.o: $(LL)/ggml/src/ggml-metal/kernels/%.metal $(HERE)/embed_metal.sh
	@mkdir -p $(dir $@)
	sh $(HERE)/embed_metal.sh $(LL) $* $(dir $@)
	$(CC) -c $(dir $@)ggml-metal-embed-$*.s -o $@

# llama, common, vendor
$(OUT)/llama/%.o: $(LL)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

define common_rule
$(OUT)/common/$(subst /,_,$(1:.cpp=.o)): $(LL)/common/$(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $$(CXXFLAGS) -c $$< -o $$@
endef
$(foreach f,$(COMMON_CXX),$(eval $(call common_rule,$(f))))
$(OUT)/common/build-info.o: $(OUT)/gen/build-info.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/vendor/httplib.o: $(LL)/vendor/cpp-httplib/httplib.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(HTTPLIB_DEFS) -c $< -o $@
$(OUT)/vendor/hash.o: $(LL)/vendor/hash/hash.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(LL)/vendor/hash -c $< -o $@
$(OUT)/vendor/xxhash.o: $(LL)/vendor/hash/xxhash/xxhash.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(LL)/vendor/hash -c $< -o $@
$(OUT)/vendor/sha1.o: $(LL)/vendor/hash/sha1/sha1.c
	@mkdir -p $(dir $@)
	$(CXX) -x c++ $(CXXFLAGS) -I$(LL)/vendor/hash -c $< -o $@
$(OUT)/vendor/sha256.o: $(LL)/vendor/hash/sha256/sha256.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(LL)/vendor/hash -c $< -o $@

# mtmd, server, cli
define mtmd_rule
$(OUT)/mtmd/$(subst /,_,$(1:.cpp=.o)): $(LL)/tools/mtmd/$(1)
	@mkdir -p $$(dir $$@)
	$(CXX) $$(CXXFLAGS) $(MTMD_DEFS) -I$(LL)/tools/mtmd -c $$< -o $$@
endef
$(foreach f,$(MTMD_CXX),$(eval $(call mtmd_rule,$(f))))

$(OUT)/server/%.o: $(LL)/tools/server/%.cpp $(OUT)/gen/ui/ui.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SERVER_INC) -c $< -o $@
$(OUT)/server/ui.o: $(OUT)/gen/ui/ui.cpp $(OUT)/gen/ui/ui.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(OUT)/gen/ui -c $< -o $@
$(OUT)/cli/%.o: $(LL)/tools/cli/%.cpp $(OUT)/gen/ui/ui.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SERVER_INC) -I$(LL)/tools/cli -c $< -o $@

# tools
# the source comes from the per-tool prerequisite lines below
$(OUT)/tools/%.o: | $(GEN_H)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(TOOL_INC) -c $(firstword $^) -o $@
$(OUT)/tools/bench-llama-bench.o: $(LL)/tools/llama-bench/llama-bench.cpp
$(OUT)/tools/bench-main.o: $(LL)/tools/llama-bench/main.cpp
$(OUT)/tools/completion.o: $(LL)/tools/completion/completion.cpp
$(OUT)/tools/completion-main.o: $(LL)/tools/completion/main.cpp
$(OUT)/tools/eval-callback.o: $(LL)/examples/eval-callback/eval-callback.cpp
$(OUT)/tools/tokenize.o: $(LL)/tools/tokenize/tokenize.cpp
$(OUT)/tools/gguf.o: $(LL)/examples/gguf/gguf.cpp
$(OUT)/tools/cli-main.o: $(LL)/tools/cli/main.cpp
$(OUT)/tools/cli-main.o: TOOL_INC := $(SERVER_INC) -I$(LL)/tools/cli
$(OUT)/tools/server-main.o: $(LL)/tools/server/main.cpp
$(OUT)/tools/server-main.o: TOOL_INC := $(SERVER_INC)
$(OUT)/tools/bench-llama-bench.o $(OUT)/tools/bench-main.o: TOOL_INC := -I$(LL)/tools/llama-bench
$(OUT)/tools/completion.o $(OUT)/tools/completion-main.o: TOOL_INC := -I$(LL)/tools/completion
$(OUT)/tools/ink-golden.o: $(PARITY)/ink_golden.cpp
$(OUT)/tools/ink-golden.o: TOOL_INC := -I$(LL)/vendor/nlohmann

$(OUT)/llama-bench: $(CORE_OBJ) $(OUT)/tools/bench-llama-bench.o $(OUT)/tools/bench-main.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-completion: $(CORE_OBJ) $(OUT)/tools/completion.o $(OUT)/tools/completion-main.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-eval-callback: $(CORE_OBJ) $(OUT)/tools/eval-callback.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-tokenize: $(CORE_OBJ) $(OUT)/tools/tokenize.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-gguf: $(GGML_OBJ) $(OUT)/tools/gguf.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-server: $(CORE_OBJ) $(HASH_OBJ) $(MTMD_OBJ) $(SERVER_OBJ) $(OUT)/tools/server-main.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/llama-cli: $(CORE_OBJ) $(HASH_OBJ) $(MTMD_OBJ) $(SERVER_OBJ) $(CLI_OBJ) $(OUT)/tools/cli-main.o
	$(CXX) $^ -o $@ $(LDFLAGS)
$(OUT)/ink-golden: $(CORE_OBJ) $(OUT)/tools/ink-golden.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OUT)/BUILD_INFO.txt: $(TOOLS)
	{ echo "llama.cpp $(TAG) $(COMMIT) dirty=$$(git -C $(LL) status --porcelain | wc -l | tr -d ' ')"; \
	  echo "variant metal(embedded)+blas(accelerate)+cpu (Scripts/parity/inkling2/llama/Makefile)"; \
	  echo "makefile sha256 $$(shasum -a 256 $(HERE)/Makefile | cut -d' ' -f1) embed_metal.sh sha256 $$(shasum -a 256 $(HERE)/embed_metal.sh | cut -d' ' -f1)"; \
	  clang --version | head -1; sw_vers -productVersion | sed 's/^/macOS /'; \
	  echo "CFLAGS $(CFLAGS) $(CPU_DEFS)"; echo "METAL_KINDS $(METAL_KINDS)"; \
	  for b in $(TOOLS) $(OUT)/ink-golden; do [ -f $$b ] && shasum -a 256 $$b; done; true; } > $@

.PHONY: all pin-check llama-cli llama-server llama-bench llama-completion llama-eval-callback llama-tokenize llama-gguf ink-golden
