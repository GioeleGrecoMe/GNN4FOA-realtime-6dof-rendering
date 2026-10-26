#!/usr/bin/env python3
"""Fail fast if release settings drift away from the legacy-Intel contract."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
build = (root / "BUILD_UNIVERSAL.command").read_text(errors="replace")
jucer = (root / "GNN4FOA.jucer").read_text(errors="replace")
pbx = (root / "Builds/MacOSX/GNN4FOA.xcodeproj/project.pbxproj").read_text(errors="replace")
corpus = "\n".join((build, jucer, pbx))

errors = []
if 'ARCHS="x86_64 arm64"' not in build:
    errors.append("build script must request x86_64 + arm64")
if 'MACOSX_DEPLOYMENT_TARGET=10.13' not in build:
    errors.append("x86_64 legacy runtime floor must remain macOS 10.13")
if 'xcodeValidArchs="x86_64,arm64"' not in jucer:
    errors.append("Projucer architecture list must contain x86_64,arm64")
if 'extraFrameworks="Accelerate"' not in jucer:
    errors.append("Projucer must link Accelerate for the vDSP backend")
if 'CLANG_X86_VECTOR_INSTRUCTIONS=default' not in build:
    errors.append("build must force Xcode Intel vector extensions to platform default")

# A 2010 Intel Mac predates AVX. Never allow explicit modern-CPU tuning in
# first-party project/build settings. Vendored Eigen may contain AVX code paths;
# they are harmless unless the compiler is explicitly allowed to select them.
unsafe = re.compile(r'(?<![A-Za-z0-9_])(?:-march=native|-mcpu=native|-mavx(?:2|512[^\s]*)?|-mfma|-mf16c)(?![A-Za-z0-9_])')
for m in unsafe.finditer(corpus):
    errors.append(f"unsafe legacy-Intel compiler flag found: {m.group(0)}")

if errors:
    for e in errors:
        print(f"ERROR: {e}")
    sys.exit(1)
print("PASS: legacy macOS/Intel policy (10.13, x86_64+arm64, no forced AVX/native CPU)")
