cmake_minimum_required(VERSION 3.4)
project(face_swap)

# Set version
set(FACE_SWAP_MAJOR_VERSION 0)
set(FACE_SWAP_MINOR_VERSION 9)
set(FACE_SWAP_PATCH_VERSION 0)
set(FACE_SWAP_VERSION ${FACE_SWAP_MAJOR_VERSION}.${FACE_SWAP_MINOR_VERSION}.${FACE_SWAP_PATCH_VERSION})

# Global configurations
# ===================================================
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
	set(DEF_INSTALL_CMAKE_DIR cmake)
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC")
	set(DEF_INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
	"Installation directory for CMake files")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_DEBUG_POSTFIX "_d" CACHE STRING
	"Add a custom \"postfix\" to static and shared libraries when in Debug build")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Optional 3rd party components
# ===================================================
option(WITH_BOOST_STATIC "Boost static libraries" OFF)
option(WITH_QT "Qt" ON)

# Build components
# ===================================================
option(BUILD_DOCS "Build documentation using Doxygen" ON)

# Find dependencies
# ===================================================

# Boost
#set(Boost_USE_STATIC_LIBS ${WITH_BOOST_STATIC})
#set(BOOST_ALL_DYN_LINK NOT ${WITH_BOOST_STATIC})
if(WIN32)
	if(${WITH_BOOST_STATIC})
		add_definitions(-DBOOST_ALL_NO_LIB)
	else()
		add_definitions(-DBOOST_ALL_DYN_LINK)
	endif()
endif()
find_package(Boost REQUIRED filesystem program_options regex timer)

# OpenCV
find_package(OpenCV REQUIRED highgui imgproc imgcodecs calib3d photo)

# face_segmentation
find_package(face_segmentation REQUIRED)

# dlib
find_package(dlib REQUIRED)

# Caffe
find_package(Caffe REQUIRED)

# Qt
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
find_package(HDF5 REQUIRED CXX)

# Eigen
find_package(Eigen3 REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# GLEW
find_package(GLEW REQUIRED)

# Doxygen
find_package(Doxygen)

# Python
find_package(PythonLibs 2.7 REQUIRED)

# Interfaces

# Add sub-directories
# ===================================================
add_subdirectory(cv_gl)
add_subdirectory(iris_sfs)
add_subdirectory(face_swap)
add_subdirectory(py_face_swap)

# Documentation
if(BUILD_DOCS)
	add_subdirectory(doc)
endif()

# Export configuration
# ===================================================

# Add all targets to the build-tree export set
set(FACE_SWAP_TARGETS face_swap iris_sfs cv_gl)
export(TARGETS ${FACE_SWAP_TARGETS}
  FILE "${PROJECT_BINARY_DIR}/face_swap-targets.cmake")
  
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE face_swap)
	 
# Create config files
configure_file(cmake/face_swap-config.cmake.in
  "${PROJECT_BINARY_DIR}/face_swap-config.cmake" @ONLY)
configure_file(cmake/face_swap-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/face_swap-config-version.cmake" @ONLY)
  
# Install config files
install(FILES
  "${PROJECT_BINARY_DIR}/face_swap-config.cmake"
  "${PROJECT_BINARY_DIR}/face_swap-config-version.cmake"
  DESTINATION "cmake" COMPONENT dev)
  
# Install the export set for use with the install-tree
install(EXPORT face_swap-targets DESTINATION cmake COMPONENT dev)

# Install sample data
file(GLOB SAMPLE_IMAGES "data/images/*.jpg")
install(FILES ${SAMPLE_IMAGES} DESTINATION data/images)
