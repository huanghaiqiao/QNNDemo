cmake_minimum_required(VERSION 3.10)
project(qnn_sample_app LANGUAGES CXX)

if(NOT DEFINED ENV{QAIRT_SDK_ROOT})
    message(FATAL_ERROR "QAIRT_SDK_ROOT is not set")
endif()

set(QAIRT_SDK_ROOT $ENV{QAIRT_SDK_ROOT})


# =========================
# 基本配置
# =========================
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 默认 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =========================
# QNN Target（等价于 Makefile 的 QNN_TARGET）
# =========================
if(ANDROID)
    set(QNN_TARGET aarch64-android)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(QNN_TARGET x86_64-linux-clang)
else()
    set(QNN_TARGET unknown)
endif()

set(TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/${QNN_TARGET})

# 输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TARGET_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TARGET_DIR})

# =========================
# 目录定义（严格对齐 Makefile）
# =========================
set(SRC_DIR src)
set(SRC_DIR_LOG src/Log)
set(SRC_DIR_UTILS src/Utils)
set(SRC_DIR_WRAPPER_UTILS src/WrapperUtils)
set(SRC_DIR_PAL_LINUX src/PAL/src/linux)
set(SRC_DIR_PAL_COMMON src/PAL/src/common)
set(PAL_INCLUDE src/PAL/include)

# QNN API include（相对路径，与你 Makefile 一致）
set(QNN_API_INCLUDE  ${QAIRT_SDK_ROOT}/include/QNN)

# =========================
# 源文件（白名单，❌ 不使用 GLOB_RECURSE）
# =========================
file(GLOB SOURCES
    ${SRC_DIR}/*.cpp
    ${SRC_DIR_LOG}/*.cpp
    ${SRC_DIR_UTILS}/*.cpp
    ${SRC_DIR_WRAPPER_UTILS}/*.cpp
    ${SRC_DIR_PAL_LINUX}/*.cpp
    ${SRC_DIR_PAL_COMMON}/*.cpp
)

# =========================
# 可执行文件
# =========================
add_executable(qnn-sample-app ${SOURCES})

# =========================
# Include 路径（等价 INCLUDES）
# =========================
target_include_directories(qnn-sample-app PRIVATE
    ${SRC_DIR}
    ${SRC_DIR_LOG}
    ${SRC_DIR_UTILS}
    ${SRC_DIR_WRAPPER_UTILS}
    ${PAL_INCLUDE}
    ${QNN_API_INCLUDE}
    ${QAIRT_SDK_ROOT}/include/QNN
)

# =========================
# 编译选项（等价 CXXFLAGS）
# =========================
# 架构相关编译参数
if(ANDROID)
    # Android arm64
    target_compile_options(qnn-sample-app PRIVATE
        -march=armv8-a
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86_64：不要强行指定 march
    # 默认就已经是 x86_64，兼容性最好
endif()


# Debug / Release 区分（等价 Makefile ifdef）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(qnn-sample-app PRIVATE
        -O0
        -g
    )
    target_compile_definitions(qnn-sample-app PRIVATE
        QNN_API=""
    )
else()
    target_compile_options(qnn-sample-app PRIVATE
        -O3
        -Wno-write-strings
        -fvisibility=hidden
    )
    target_compile_definitions(qnn-sample-app PRIVATE QNN_API_EXPORT)

endif()

# =========================
# 链接选项（等价 LDFLAGS / LIBS）
# =========================

target_link_libraries(qnn-sample-app PRIVATE dl)

if(NOT ANDROID)
    target_link_libraries(qnn-sample-app PRIVATE pthread)
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(qnn-sample-app PRIVATE
        -fvisibility=hidden
        -flto
    )
endif()
