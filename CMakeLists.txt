cmake_minimum_required(VERSION 3.16)
project(cppbone VERSION 0.2.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB_RECURSE NETBONE_SOURCES CONFIGURE_DEPENDS netbone/*.cpp)
add_library(netbone STATIC ${NETBONE_SOURCES})
target_include_directories(netbone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# A wheel build (scikit-build-core sets SKBUILD) only needs the Python module.
if(NOT SKBUILD)
  add_executable(netbone_example examples/example.cpp)
  target_link_libraries(netbone_example PRIVATE netbone)

  enable_testing()
  add_executable(test_netbone tests/test_netbone.cpp)
  target_link_libraries(test_netbone PRIVATE netbone)
  add_test(NAME test_netbone COMMAND test_netbone ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv)
endif()

# Python module `netbone_cpp`, off by default: cmake -DNETBONE_PYTHON=ON [-DPython_EXECUTABLE=...]
option(NETBONE_PYTHON "Build the netbone_cpp Python module" OFF)
if(NETBONE_PYTHON)
  set_target_properties(netbone PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
  pybind11_add_module(netbone_cpp python/netbone_py.cpp)
  target_link_libraries(netbone_cpp PRIVATE netbone)
  install(TARGETS netbone_cpp LIBRARY DESTINATION .)
  if(NOT SKBUILD)
    add_test(NAME test_python COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py
             ${CMAKE_CURRENT_SOURCE_DIR}/data/data.csv)
    set_tests_properties(test_python PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:netbone_cpp>")
  endif()
endif()
