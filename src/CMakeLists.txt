cmake_minimum_required(VERSION 2.8)

project(cgcv)

if (NOT WIN32)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
endif()

# we need one submit target per task:
cmake_policy(SET CMP0002 OLD)

# compile full version (1) or student version (0):
set(FULL_VERSION 1 CACHE STRING "Full or student version")

# Final run, using generated reference solution in each task:
set(FINAL_RUN 0 CACHE STRING "Final run")

# compile with bonus or not:
set(BONUS 0 CACHE STRING "With or without bonus")

# Generate reference solution:
set(GENERATE_REF 0 CACHE STRING "Generate reference solution")

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."   FORCE)
endif()

set(SOURCE_WILDCARDS *.h *.H *.hpp *.hh *.hxx *.c *.C *.cpp *.cc *.cxx)

macro(ADD_SUBDIRECTORY_IF_EXISTS dir)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    add_subdirectory(${dir})
  endif(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endmacro()

MACRO(add_pch header target sources)
  if(MSVC)
    foreach (src_file ${${sources}})
      set_source_files_properties(
        ${src_file}
        PROPERTIES
        COMPILE_FLAGS "/Yu${header}"
      )
    endforeach (src_file ${${sources}})

    set_source_files_properties(
      ${target}
      PROPERTIES
      COMPILE_FLAGS "/Yc${header}"
    )

    LIST(APPEND ${sources} ${target})
  endif()
ENDMACRO()


# For OpenCV support at pluto.tugraz.at
include_directories(/software/opencv/include)

add_definitions(-DFULL_VERSION=${FULL_VERSION})
add_definitions(-DFINAL_RUN=${FINAL_RUN})
add_definitions(-DGENERATE_REF=${GENERATE_REF})

if (BONUS)
    add_definitions(-DBONUS)
endif()

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
endif()

option(BUILD_CV1 "Build CV1" ON)

if (BUILD_CV1)
  add_subdirectory_if_exists(cv)
endif()

add_custom_target(clean-all COMMAND make clean \; find \\\( -name CMakeFiles -or -name CMakeCache.txt -or -name Makefile -or -name cmake_install.cmake -or -name progress.make -or -name _CPack_Packages -or -name '*~' \\\) -prune -exec rm -rf {} '\;')
