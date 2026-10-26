cmake_minimum_required(VERSION 3.16)
project(cppbone VERSION 0.2.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB_RECURSE CPPBONE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(cppbone STATIC ${CPPBONE_SOURCES})
# Public headers in include/cppbone/; src/ holds the sources and the private detail.hpp.
target_include_directories(cppbone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# OpenMP parallelises the per-node / per-edge / per-source loops; without it everything runs serially.
option(CPPBONE_OPENMP "Build with OpenMP when the compiler supports it" ON)
if(CPPBONE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cppbone PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found: cppbone will run single-threaded")
  endif()
endif()

# A wheel build (scikit-build-core sets SKBUILD) only needs the Python module.
if(NOT SKBUILD)
  # Command-line tool: argument parsing (Taywee/args) and progress bars (p-ranav/indicators), both header-only.
  include(FetchContent)
  set(ARGS_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(ARGS_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(ARGS_ENABLE_SECURITY_HARDENING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(args
    URL https://github.com/Taywee/args/archive/refs/tags/6.6.1.tar.gz
    URL_HASH SHA256=61cdbb13f55493e6ffe07018c86b17f9ff68a5c1059eb15f4f079e04429915ff
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_Declare(indicators
    URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
    URL_HASH SHA256=70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(args indicators)

  add_executable(cppbone_cli apps/cppbone_cli.cpp)
  target_link_libraries(cppbone_cli PRIVATE cppbone taywee::args indicators::indicators)

  enable_testing()
  add_executable(test_cppbone tests/test_cppbone.cpp)
  target_link_libraries(test_cppbone PRIVATE cppbone)
  add_test(NAME test_cppbone COMMAND test_cppbone ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv)
  # Regression suite: every function and method/filter combination on data/data.csv, checked against the C++
  # snapshot and the original NetBone results in tests/regression/baseline (see tests/regression/README.md).
  add_executable(cppbone_regression tests/regression/regression.cpp)
  target_link_libraries(cppbone_regression PRIVATE cppbone)
  add_test(NAME regression COMMAND cppbone_regression ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv
           ${CMAKE_CURRENT_SOURCE_DIR}/tests/regression)
  # `cmake --build <dir> --target regression_baseline` rewrites the C++ snapshot after an intended change.
  add_custom_target(regression_baseline
    COMMAND cppbone_regression ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv ${CMAKE_CURRENT_SOURCE_DIR}/tests/regression
            --write-cpp-baseline
    DEPENDS cppbone_regression
    COMMENT "Writing tests/regression/baseline/cpp")
  add_test(NAME test_cli COMMAND cppbone_cli ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv -q -m all -f fraction -v 0.1
           -M all --ks all --consensus --progression 0.05,0.1,0.2 -o ${CMAKE_CURRENT_BINARY_DIR}/cli_out)
endif()

# Boost.Graph and Boost.Math (header-only parts): the graph algorithms (shortest paths, betweenness, spanning tree,
# planarity, components) and the binomial distribution. A system Boost if found, else the official release.
find_package(Boost 1.80 CONFIG QUIET COMPONENTS headers)
if(Boost_FOUND)
  target_link_libraries(cppbone PRIVATE Boost::headers)
  message(STATUS "cppbone: using Boost ${Boost_VERSION}")
else()
  message(STATUS "cppbone: no system Boost, fetching Boost 1.90 (Graph and Math only)")
  include(FetchContent)
  set(BOOST_INCLUDE_LIBRARIES graph math CACHE STRING "" FORCE)
  set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
  FetchContent_Declare(Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
    URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(Boost)
  target_link_libraries(cppbone PRIVATE Boost::graph Boost::math)
endif()

# LAPACK for the dense solves (the ECM Hessian): a system LAPACK (OpenBLAS, MKL, Accelerate, netlib; choose with
# -DBLA_VENDOR=...), else OpenBLAS downloaded and built here (a few minutes).
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  target_link_libraries(cppbone PRIVATE LAPACK::LAPACK)
  message(STATUS "cppbone: using LAPACK ${LAPACK_LIBRARIES}")
else()
  message(STATUS "cppbone: no system LAPACK, building OpenBLAS 0.3.33")
  include(FetchContent)
  # C_LAPACK: LAPACK from its C translation, so no Fortran compiler is needed.
  set(C_LAPACK ON CACHE BOOL "" FORCE)
  set(NOFORTRAN ON CACHE BOOL "" FORCE)
  set(BUILD_WITHOUT_LAPACKE ON CACHE BOOL "" FORCE)
  # Only LAPACK's Fortran interface (dposv_) is used. Without CBLAS, OpenBLAS also skips its utest/ctest folders,
  # which would otherwise register tests in this project's ctest that are never built.
  set(NO_CBLAS ON CACHE BOOL "" FORCE)
  set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(USE_OPENMP ${OpenMP_CXX_FOUND} CACHE BOOL "" FORCE)  # share the OpenMP runtime instead of OpenBLAS's pthreads
  # A wheel runs on other machines: include kernels for every CPU and pick one at runtime, instead of tuning for the
  # build machine (MSVC builds of OpenBLAS do not support this).
  if(SKBUILD AND NOT MSVC)
    set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)
  endif()
  FetchContent_Declare(openblas
    URL https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.33.tar.gz
    URL_HASH SHA256=6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(openblas)
  set_target_properties(openblas_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(cppbone PRIVATE openblas_static)
endif()

# Python module `cppbone`, off by default: cmake -DCPPBONE_PYTHON=ON [-DPython_EXECUTABLE=...]
option(CPPBONE_PYTHON "Build the cppbone Python module" OFF)
if(CPPBONE_PYTHON)
  set_target_properties(cppbone PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(PYBIND11_FINDPYTHON ON)
  # Wheel builds get pybind11 from build-system.requires; a plain CMake build downloads it.
  find_package(pybind11 3 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
      URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()
  # The target needs its own name (cppbone is the library); the module file is still cppbone.<ext>.
  pybind11_add_module(cppbone_python python/cppbone_py.cpp)
  set_target_properties(cppbone_python PROPERTIES OUTPUT_NAME cppbone)
  target_link_libraries(cppbone_python PRIVATE cppbone)
  install(TARGETS cppbone_python LIBRARY DESTINATION .)
  if(NOT SKBUILD)
    add_test(NAME test_python COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py
             ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv)
    set_tests_properties(test_python PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:cppbone_python>")
  endif()
endif()
