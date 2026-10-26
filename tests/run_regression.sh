#!/usr/bin/env bash
# Builds the regression runner and checks every case on data/data.csv against tests/regression/baseline.
#
#   tests/run_regression.sh                      build, then run every case and print the result of each
#   tests/run_regression.sh --only method.lans   only the cases whose name contains "method.lans"
#   tests/run_regression.sh --quiet              print only the failing cases and the summary
#   tests/run_regression.sh --update             rewrite the C++ baseline after an intended change
#
# The build goes to build-regression/ (override with BUILD_DIR=...). Exit status: 0 when every case passes.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$root/build-regression}"

verbose=--verbose
args=()
for a in "$@"; do
  case "$a" in
    --update) args+=("--write-cpp-baseline") ;;
    --quiet) verbose= ;;
    *) args+=("$a") ;;
  esac
done
[[ -n "$verbose" ]] && args+=("$verbose")

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -Wno-deprecated > /dev/null
cmake --build "$build" --target cppbone_regression --config Release -j > /dev/null

# Single-config generators (Make, Ninja) put the runner in the build directory, Visual Studio in Release/.
runner="$build/cppbone_regression"
for candidate in "$build/cppbone_regression" "$build/Release/cppbone_regression.exe" "$build/cppbone_regression.exe"; do
  if [[ -x "$candidate" ]]; then runner="$candidate"; break; fi
done

exec "$runner" "$root/data/data.csv" "$root/tests/regression" ${args[@]+"${args[@]}"}
