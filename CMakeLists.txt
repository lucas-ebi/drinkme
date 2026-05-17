cmake_minimum_required(VERSION 3.15)
project(drinkme CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use an installed Eigen3 if available; otherwise pull it in automatically.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    )
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

add_executable(drinkme src/main.cpp)
target_include_directories(drinkme PRIVATE include)
target_link_libraries(drinkme Eigen3::Eigen)
target_compile_options(drinkme PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
