# CMakeLists.txt for kws_sdk library

# Define the library target (STATIC or SHARED)
add_library(kws_sdk STATIC) # Use STATIC for easier linking in the demo initially

# List the source files for the library
target_sources(kws_sdk PRIVATE
    kws_sdk.cpp
    mfcc.cpp
    gmm_hmm.cpp
    model_loader.cpp
    # Add any other .cpp files belonging to the SDK implementation
)

# Specify public include directories (needed by users of the library)
# $<INSTALL_INTERFACE:include> is for when installing the library
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> is for building locally
target_include_directories(kws_sdk PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Specify private include directories (only needed for compiling the library itself)
# For example, if third-party headers are only included in .cpp files
target_include_directories(kws_sdk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} # Allows including mfcc.h etc. from kws_sdk.cpp
    # Explicitly add include directories for dependencies
    ${FFTW_INCLUDE_DIRS}        # From pkg-config or find_package
    ${EIGEN3_INCLUDE_DIRS}      # From pkg-config or find_package
    # nlohmann_json is usually header-only and its include path might be
    # managed by linking the nlohmann_json::nlohmann_json target, but explicit
    # addition might be needed in some cases (though less likely).
    # ${nlohmann_json_INCLUDE_DIRS} # If needed
)

# Link dependencies
# PUBLIC: If SDK headers use types/functions from the dependency
# PRIVATE: If the dependency is only used within the SDK's .cpp files
target_link_libraries(kws_sdk PRIVATE
    # Link math library (needed for std::sqrt, std::log etc.)
    m # On Linux/macOS, often implicitly linked but good practice

    # Link FFTW (using the target determined in the top-level CMakeLists.txt)
    # Use the variable set in the parent scope which holds library names or target
    ${FFTW_LINK_TARGET}

    # Link Eigen (usually header-only, but link if necessary)
    # No standard CMake target for Eigen, linking often not needed.
    # If you encounter linking errors related to Eigen, add link targets here.
    # Check if EIGEN3_LIBRARIES is set by pkg-config and link if necessary
    # ${EIGEN3_LIBRARIES}

    # Link nlohmann_json (target name from FetchContent or find_package)
    # Use the modern target name if available
    nlohmann_json::nlohmann_json

    # Link other necessary libraries
)

# Set C++ standard for the library
target_compile_features(kws_sdk PRIVATE cxx_std_17) # Requires C++17 for std::filesystem

# --- Installation (Optional but good practice) ---
# This part defines how to install the library and its header files
# include(GNUInstallDirs)
# install(TARGETS kws_sdk
#         EXPORT kws_sdk_targets
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # For SHARED libs on Windows
#         INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) # For header-only part if any

# install(FILES ../include/kws_sdk.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kws) # Install public header

# install(EXPORT kws_sdk_targets
#         FILE kws_sdk_targets.cmake
#         NAMESPACE kws::
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kws_sdk)

# # Create a config file for find_package()
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     kws_sdk_config_version.cmake
#     VERSION ${PROJECT_VERSION} # Define PROJECT_VERSION in top-level CMakeLists
#     COMPATIBILITY SameMajorVersion)

# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/kws_sdk_config_version.cmake
#     # kws_sdk_config.cmake # You might need to create this file
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kws_sdk) 