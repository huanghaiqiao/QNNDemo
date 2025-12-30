cd build
cmake ..
make

cd build
cmake ..   -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a   -DANDROID_PLATFORM=android-21   -DCMAKE_BUILD_TYPE=Release
