#include "build-info.h"

#include <cstdio>
#include <string>

int LLAMA_BUILD_NUMBER = 0;
char const * LLAMA_COMMIT = "b1b770d9";
char const * LLAMA_COMPILER = "clang";
char const * LLAMA_BUILD_TARGET = "arm64-apple-darwin-metal";

int llama_build_number(void) {
    return LLAMA_BUILD_NUMBER;
}

const char * llama_commit(void) {
    return LLAMA_COMMIT;
}

const char * llama_compiler(void) {
    return LLAMA_COMPILER;
}

const char * llama_build_target(void) {
    return LLAMA_BUILD_TARGET;
}

const char * llama_build_info(void) {
    static std::string s = "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
    return s.c_str();
}

void llama_print_build_info(const char * llama_version, FILE * stream) {
    fprintf(stream, "version: %s (build %d, commit %s)\n", llama_version, llama_build_number(), llama_commit());
    fprintf(stream, "built with %s for %s\n", llama_compiler(), llama_build_target());
}
