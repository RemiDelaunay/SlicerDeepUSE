cmake_minimum_required(VERSION 3.13.4)

project(DeepUSE)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DeepUSE")
set(EXTENSION_CATEGORY "DeepUSE")
set(EXTENSION_CONTRIBUTORS "Remi Delaunay (UCL,KCL)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/DeepUSE.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/DeepUSE/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeepUSE)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
