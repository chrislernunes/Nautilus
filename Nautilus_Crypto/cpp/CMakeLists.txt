cmake_minimum_required(VERSION 3.18)
project(nautilus_cpp VERSION 1.0.0 LANGUAGES CXX)

# ── Compiler standards & flags ───────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive optimisation (match what Citadel prod would use)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(
    -O3
    -march=native          # enable AVX2/AVX-512 if available
    -ffast-math
    -funroll-loops
    -DNDEBUG
  )
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────

# pybind11
find_package(pybind11 REQUIRED)

# libwebsockets (optional — Python WS bridge used if not present)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LWS libwebsockets)
endif()
if(LWS_FOUND)
  message(STATUS "libwebsockets found — full C++ WS client enabled")
  add_compile_definitions(HAVE_LIBWEBSOCKETS)
else()
  message(WARNING "libwebsockets NOT found — Python WS bridge will be used")
endif()

# simdjson (optional — fast JSON parsing)
find_package(simdjson QUIET)
if(simdjson_FOUND)
  message(STATUS "simdjson found — zero-copy JSON parsing enabled")
  add_compile_definitions(HAVE_SIMDJSON)
else()
  message(WARNING "simdjson NOT found — using built-in minimal JSON parser")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(CPP_SOURCES
  src/websocket_client.cpp
  src/bindings.cpp
)

# ── pybind11 module ──────────────────────────────────────────────────────────
pybind11_add_module(nautilus_cpp ${CPP_SOURCES})

target_include_directories(nautilus_cpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link optional dependencies
if(LWS_FOUND)
  target_include_directories(nautilus_cpp PRIVATE ${LWS_INCLUDE_DIRS})
  target_link_libraries(nautilus_cpp PRIVATE ${LWS_LIBRARIES})
endif()

if(simdjson_FOUND)
  target_link_libraries(nautilus_cpp PRIVATE simdjson::simdjson)
endif()

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(nautilus_cpp PRIVATE Threads::Threads)

# ── Output location ──────────────────────────────────────────────────────────
# Place the .so next to the Python core package for import convenience
set_target_properties(nautilus_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../python/core
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS nautilus_cpp
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../python/core
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags:  ${CMAKE_CXX_FLAGS}")
