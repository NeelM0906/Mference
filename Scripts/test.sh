#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."

developer_directory="${DEVELOPER_DIR:-$(xcode-select -p)}"
framework_directory="$developer_directory/Library/Developer/Frameworks"
library_directory="$developer_directory/Library/Developer/usr/lib"
if [[ -d "$framework_directory/Testing.framework" ]]; then
  set -- -Xswiftc -F -Xswiftc "$framework_directory" -Xlinker -F -Xlinker "$framework_directory" -Xlinker -rpath -Xlinker "$framework_directory" "$@"
  if [[ -f "$library_directory/lib_TestingInterop.dylib" ]]; then
    set -- -Xlinker -rpath -Xlinker "$library_directory" "$@"
  fi
fi

exec swift test --no-parallel "$@"
