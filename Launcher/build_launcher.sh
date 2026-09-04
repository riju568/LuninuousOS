#!/bin/bash
MAX_ATTEMPTS=3
ATTEMPT=1
SUCCESS=0

# Ensure entry point exists as main.py
if [ -f "LuninuousLauncher.py" ]; then
    cp LuninuousLauncher.py main.py
    echo "[Init] Copied LuninuousLauncher.py to main.py"
fi

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo " [BUILD PIPELINE] Attempt $ATTEMPT of $MAX_ATTEMPTS"

    buildozer android debug

    if [ $? -eq 0 ]; then
        echo "SUCCESS: APK compiled successfully on attempt $ATTEMPT!"
        SUCCESS=1
        break
    else
        echo "WARNING: Build failed on attempt $ATTEMPT. Triggering exception recovery..."
        if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo "🔧 [Self-Healing] Purging build cache and resetting environment..."
            buildozer clean
            rm -rf .buildozer/android/platform/build-*
        fi
    fi

    ATTEMPT=$((ATTEMPT + 1))
done

if [ $SUCCESS -eq 0 ]; then
    echo "FATAL ERROR: All 3 build attempts failed. Check toolchain dependencies (JDK, Cython, NDK)."
    exit 1
fi