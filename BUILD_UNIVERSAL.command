#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
PROJ="$ROOT/Builds/MacOSX/GNN4FOA.xcodeproj"
OUT="$ROOT/Builds/ReleaseUniversal_v2"
DIST="$ROOT/GNN4FOA_v2_macOS_Universal"
LOCAL_JUCE="$ROOT/ThirdParty/JUCE"
JUCE_TAG="9.0.1"

check_juce_tree () {
  local j="$1"
  [ -d "$j/modules" ] || return 1
  [ -f "$j/modules/juce_graphics/juce_graphics_lunasvg.c" ] || return 1
  [ -f "$j/modules/juce_graphics/juce_graphics_Sheenbidi.c" ] || return 1
  [ -f "$j/modules/juce_gui_basics/juce_gui_basics_5.cpp" ] || return 1
  [ -f "$j/modules/juce_audio_processors_headless/juce_audio_processors_headless.h" ] || return 1
  [ -f "$j/modules/juce_audio_plugin_client/VST3/juce_VST3ManifestHelper.cpp" ] || return 1
  return 0
}

prepare_juce () {
  mkdir -p "$ROOT/ThirdParty"

  if check_juce_tree "$LOCAL_JUCE"; then
    echo "JUCE: using project-local tree: $LOCAL_JUCE"
    return
  fi

  if [ -e "$LOCAL_JUCE" ] || [ -L "$LOCAL_JUCE" ]; then
    rm -rf "$LOCAL_JUCE"
  fi

  if [ -n "${JUCE_ROOT:-}" ]; then
    if check_juce_tree "$JUCE_ROOT"; then
      echo "JUCE: using JUCE_ROOT=$JUCE_ROOT"
      ln -s "$JUCE_ROOT" "$LOCAL_JUCE"
      return
    fi
    echo "ERROR: JUCE_ROOT is set but is not compatible with the generated JUCE 9 wrappers:"
    echo "       $JUCE_ROOT"
    echo "Missing at least one of: juce_graphics_lunasvg.c, juce_gui_basics_5.cpp,"
    echo "juce_audio_processors_headless, or the VST3 manifest helper."
    exit 10
  fi

  command -v git >/dev/null 2>&1 || {
    echo "ERROR: a compatible JUCE tree was not found and git is unavailable."
    echo "Install Xcode/Command Line Tools, or run:"
    echo "  JUCE_ROOT=/path/to/JUCE bash BUILD_UNIVERSAL.command"
    exit 11
  }

  echo "JUCE: compatible local modules not found."
  echo "JUCE: cloning exact tag $JUCE_TAG into ThirdParty/JUCE ..."
  git clone --depth 1 --branch "$JUCE_TAG" https://github.com/juce-framework/JUCE.git "$LOCAL_JUCE"

  if ! check_juce_tree "$LOCAL_JUCE"; then
    echo "ERROR: JUCE $JUCE_TAG was downloaded but the expected JUCE 9 module layout is missing."
    exit 12
  fi
}

command -v xcodebuild >/dev/null 2>&1 || {
  echo "ERROR: xcodebuild not found. Install full Xcode and select it with xcode-select."
  exit 13
}

XCODE_VERSION="$(xcodebuild -version 2>/dev/null | awk 'NR==1 {print $2}')"
XCODE_MAJOR="${XCODE_VERSION%%.*}"
XCODE_REST="${XCODE_VERSION#*.}"
XCODE_MINOR="${XCODE_REST%%.*}"
if [[ -z "$XCODE_VERSION" ]] || (( XCODE_MAJOR < 12 )) || { (( XCODE_MAJOR == 12 )) && (( XCODE_MINOR < 4 )); }; then
  echo "ERROR: JUCE 9.0.1 requires Xcode 12.4 or newer to build this package reliably."
  echo "The build machine may be newer than the target: generated binaries still target macOS 10.13 and include x86_64 for Mojave 10.14.6."
  echo "Detected Xcode: ${XCODE_VERSION:-unknown}"
  exit 15
fi
echo "Xcode: $XCODE_VERSION (JUCE 9.0.1 build requirement satisfied)"

DEV_DIR="$(xcode-select -p 2>/dev/null || true)"
if [[ "$DEV_DIR" == "/Library/Developer/CommandLineTools"* ]]; then
  echo "ERROR: xcodebuild is using Command Line Tools instead of full Xcode."
  echo "If Xcode is installed in /Applications/Xcode.app, run:"
  echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
  exit 14
fi

UNSAFE_PLUGIN_MACRO_RE='JucePlugin_(Name|Desc)=\\\\\\"[^\"]* [^\"]*\\\\\\"'
if grep -Eq "$UNSAFE_PLUGIN_MACRO_RE" "$PROJ/project.pbxproj"; then
  echo "ERROR: unsafe plug-in Name/Desc preprocessor definition contains a literal space."
  echo "This can be split by Xcode response files and accidentally define tokens such as v1."
  echo "Use a host-safe compile identity such as GNN4FOA_v2."
  exit 16
fi
echo "Plug-in compile identity: host-safe (no whitespace in Name/Desc preprocessor macros)"

# The wrapper targets must link the renamed Shared Code static library.
# A stale -lGNN4FOA here causes a late AU/VST3 linker failure even though
# libGNN4FOA_v2.a was built successfully.
if grep -Eq 'OTHER_LDFLAGS = .* -lGNN4FOA(_v1)?([";[:space:]]|$)' "$PROJ/project.pbxproj"; then
  echo "ERROR: stale wrapper linker flag from GNN4FOA/v1 found."
  echo "The v2 project builds libGNN4FOA_v2.a and every wrapper must link -lGNN4FOA_v2."
  exit 17
fi
LINK_COUNT="$(grep -Ec 'OTHER_LDFLAGS = .* -lGNN4FOA_v2([";[:space:]]|$)' "$PROJ/project.pbxproj" || true)"
if (( LINK_COUNT < 6 )); then
  echo "ERROR: expected all v2 wrapper configurations to link -lGNN4FOA_v2; found only $LINK_COUNT linker settings."
  exit 18
fi
if ! grep -q 'path = "libGNN4FOA_v2.a"' "$PROJ/project.pbxproj"; then
  echo "ERROR: Shared Code product is not libGNN4FOA_v2.a."
  exit 19
fi
echo "Shared Code link identity: libGNN4FOA_v2.a <-> -lGNN4FOA_v2 ($LINK_COUNT wrapper configs)"

command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is required for the first-party source integrity gate."
  exit 20
}
echo "Running fail-fast source/backend/legacy compatibility gates before Xcode..."
bash "$ROOT/Tests/portable/run_release_gate.sh"

prepare_juce

# Purge the project-local Xcode build database as well as distributable outputs.
# This removes stale product records from the earlier whitespace-name build.
rm -rf "$ROOT/Builds/MacOSX/build" "$OUT" "$DIST"
mkdir -p "$OUT" "$DIST/Payload"

echo "Building GNN4FOA_v2 AU + VST3 (x86_64 + arm64, min macOS 10.13)..."
echo "JUCE modules: $LOCAL_JUCE/modules"
# Keep the Intel slice on the conservative x86_64 baseline used by legacy
# 2010-era Macs. In particular, never let a project/user setting silently turn
# on SSE4.2/AVX merely because the build host is newer. Apple Silicon keeps its
# normal arm64 code generation.
COMMON=( -project "$PROJ" -configuration Release ARCHS="x86_64 arm64" ONLY_ACTIVE_ARCH=NO MACOSX_DEPLOYMENT_TARGET=10.13 CLANG_X86_VECTOR_INSTRUCTIONS=default GCC_ENABLE_SSE3_EXTENSIONS=NO GCC_ENABLE_SUPPLEMENTAL_SSE3_INSTRUCTIONS=NO GCC_ENABLE_SSE41_EXTENSIONS=NO GCC_ENABLE_SSE42_EXTENSIONS=NO CONFIGURATION_BUILD_DIR="$OUT" )
xcodebuild "${COMMON[@]}" -target "GNN4FOA - AU" build
xcodebuild "${COMMON[@]}" -target "GNN4FOA - VST3" build

AU="$OUT/GNN4FOA_v2.component"
VST="$OUT/GNN4FOA_v2.vst3"
[ -d "$AU" ] || { echo "ERROR: AU not found at $AU"; exit 2; }
[ -d "$VST" ] || { echo "ERROR: VST3 not found at $VST"; exit 2; }

ditto "$AU" "$DIST/Payload/GNN4FOA_v2.component"
ditto "$VST" "$DIST/Payload/GNN4FOA_v2.vst3"

verify_bin () {
  local b="$1"
  echo "-- $b"
  lipo -info "$b"
  if ! lipo -archs "$b" | grep -q 'x86_64'; then echo "ERROR: x86_64 missing"; exit 3; fi
  if ! lipo -archs "$b" | grep -q 'arm64'; then echo "ERROR: arm64 missing"; exit 3; fi
  echo "deployment target:"
  otool -l "$b" | grep -E -A4 'LC_VERSION_MIN_MACOSX|LC_BUILD_VERSION' | grep -E 'version| minos ' | head -2 || true
  if ! otool -l "$b" | grep -E -A4 'LC_VERSION_MIN_MACOSX|LC_BUILD_VERSION' | grep -Eq 'version[[:space:]]+10\.13|minos[[:space:]]+10\.13'; then
    echo "ERROR: binary does not advertise the required macOS 10.13 deployment target"
    exit 5
  fi
  if strings "$b" | grep -q '\$NEWLAPACK'; then echo "ERROR: NEWLAPACK symbol still present"; exit 4; fi
  if strings "$b" | grep -q 'cblas_'; then echo "ERROR: CBLAS symbol still present"; exit 4; fi
  if ! strings "$b" | grep -q 'vDSP_mmul'; then echo "WARNING: vDSP_mmul string not found; inspect with nm/otool"; fi

  # 2010 Intel Macs do not provide AVX. Inspect only the x86_64 slice and fail
  # if the compiler emitted VEX/AVX-family instructions despite the conservative
  # build settings. This is a runtime-safety gate, not merely a project audit.
  local thin="$OUT/.legacy-check-$(basename "$b")-x86_64"
  lipo "$b" -thin x86_64 -output "$thin"
  if otool -tvV "$thin" | awk '{print $2}' | grep -Eq '^v[a-zA-Z0-9_.]+$'; then
    echo "ERROR: x86_64 slice contains AVX/VEX-style instructions; unsafe for 2010-era Intel Macs"
    otool -tvV "$thin" | awk '$2 ~ /^v[a-zA-Z0-9_.]+$/ {print; if (++n == 12) exit}'
    rm -f "$thin"
    exit 6
  fi
  rm -f "$thin"
  echo "legacy Intel CPU gate: PASS (no AVX/VEX instruction mnemonics)"
}
verify_bin "$DIST/Payload/GNN4FOA_v2.component/Contents/MacOS/GNN4FOA_v2"
verify_bin "$DIST/Payload/GNN4FOA_v2.vst3/Contents/MacOS/GNN4FOA_v2"

codesign --force --deep --sign - "$DIST/Payload/GNN4FOA_v2.component"
codesign --force --deep --sign - "$DIST/Payload/GNN4FOA_v2.vst3"

cat > "$DIST/install.command" <<'EOS'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "$HOME/Library/Audio/Plug-Ins/Components" "$HOME/Library/Audio/Plug-Ins/VST3"
rm -rf "$HOME/Library/Audio/Plug-Ins/Components/GNN4FOA_v2.component"
rm -rf "$HOME/Library/Audio/Plug-Ins/VST3/GNN4FOA_v2.vst3"
ditto "Payload/GNN4FOA_v2.component" "$HOME/Library/Audio/Plug-Ins/Components/GNN4FOA_v2.component"
ditto "Payload/GNN4FOA_v2.vst3" "$HOME/Library/Audio/Plug-Ins/VST3/GNN4FOA_v2.vst3"
xattr -dr com.apple.quarantine "$HOME/Library/Audio/Plug-Ins/Components/GNN4FOA_v2.component" 2>/dev/null || true
xattr -dr com.apple.quarantine "$HOME/Library/Audio/Plug-Ins/VST3/GNN4FOA_v2.vst3" 2>/dev/null || true
killall -9 AudioComponentRegistrar 2>/dev/null || true
killall -9 auvaltool 2>/dev/null || true
echo "Installed GNN4FOA_v2. Restart your DAW and rescan plug-ins if needed."
EOS
chmod +x "$DIST/install.command"
cat > "$DIST/verify.command" <<'EOS'
#!/bin/bash
set -euo pipefail
for B in "$HOME/Library/Audio/Plug-Ins/Components/GNN4FOA_v2.component/Contents/MacOS/GNN4FOA_v2" "$HOME/Library/Audio/Plug-Ins/VST3/GNN4FOA_v2.vst3/Contents/MacOS/GNN4FOA_v2"; do
  echo "== $B =="
  [ -f "$B" ] || { echo "MISSING"; continue; }
  lipo -info "$B"
  echo "backend symbols:"
  nm -u "$B" 2>/dev/null | grep -E 'vDSP_mmul|cblas_|NEWLAPACK' || true
  strings "$B" | grep -E 'GNN4FOA-V2.0.0-ISPL|GNN4FOA_v2' | head || true
  codesign --verify --deep --strict --verbose=2 "$(dirname "$(dirname "$B")")" 2>&1 || true
  echo
done
EOS
chmod +x "$DIST/verify.command"

ZIP="$ROOT/GNN4FOA_v2.0.0_macOS_Universal_installer.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$DIST" "$ZIP"
shasum -a 256 "$ZIP" > "$ZIP.sha256"
echo "DONE: $ZIP"
cat "$ZIP.sha256"
