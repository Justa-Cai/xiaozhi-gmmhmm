# CMakeLists.txt (Top Level: sdk/)
cmake_minimum_required(VERSION 3.16) # Increased for FetchContent details if needed
project(KWS_SDK_Project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++

# --- User Options ---
# Option to build the demo application
option(KWS_SDK_BUILD_DEMO "Build the demo application" ON)
# Option to use system libraries vs FetchContent (Example for PortAudio)
option(KWS_SDK_PREFER_SYSTEM_LIBS "Prefer system libraries if found" ON)

# --- Find/Declare Dependencies ---
include(FetchContent) # Needed for fetching dependencies

# -- PortAudio --
set(PORTAUDIO_FETCH_GIT_REPOSITORY https://github.com/PortAudio/portaudio.git)
set(PORTAUDIO_FETCH_GIT_TAG v19.7.0) # Use a specific stable tag

if(KWS_SDK_PREFER_SYSTEM_LIBS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
    endif()
endif()

if(NOT PORTAUDIO_FOUND)
    message(STATUS "PortAudio not found via PkgConfig or system search disabled. Fetching from source.")
    FetchContent_Declare(
        portaudio_fetch # Name for FetchContent internal management
        GIT_REPOSITORY ${PORTAUDIO_FETCH_GIT_REPOSITORY}
        GIT_TAG ${PORTAUDIO_FETCH_GIT_TAG}
        # Add configure/build commands if the dependency doesn't use CMake well itself
        # Or specify CMAKE_ARGS if it's a CMake project
        CMAKE_ARGS -DPA_BUILD_STATIC=ON # Example: Build static lib
                   -DPA_BUILD_SHARED=OFF
                   -DPA_ENABLE_DEBUG_OUTPUT=OFF
                   # Add other configure options as needed
    )
    # Make the content available (downloads and potentially configures/builds)
    FetchContent_MakeAvailable(portaudio_fetch)

    # Check if FetchContent successfully created the target 'portaudio_static'
    # The target name might vary based on the fetched project's CMakeLists.txt
    # You might need to inspect the build directory after fetching to confirm the target name.
    if(TARGET portaudio_static)
         # Define variables expected by find_package manually for consistency
         set(PORTAUDIO_FOUND TRUE)
         # Try to get include directory from target property if possible
         get_target_property(PORTAUDIO_INCLUDE_DIRS portaudio_static INTERFACE_INCLUDE_DIRECTORIES)
         if(NOT PORTAUDIO_INCLUDE_DIRS)
             # Fallback if property is not set
             set(PORTAUDIO_INCLUDE_DIRS ${portaudio_SOURCE_DIR}/include)
         endif()
         set(PORTAUDIO_LIBRARIES portaudio_static) # Use the target name
         message(STATUS "PortAudio fetched and target 'portaudio_static' created.")
    else()
         message(WARNING "FetchContent for PortAudio finished, but target 'portaudio_static' not found. Linking might fail.")
         # Attempt to link directly to the include directory if FetchContent didn't create a target
         # This is less reliable
         set(PORTAUDIO_INCLUDE_DIRS ${portaudio_SOURCE_DIR}/include) # Adjust path if needed
         # No library target available, linking might need manual specification later
         message(WARNING "Linking against fetched PortAudio might require manual setup in demo/CMakeLists.txt")
         # Consider setting PORTAUDIO_FOUND to FALSE here if demo relies on the target existing
         # set(PORTAUDIO_FOUND FALSE)
    endif()

else()
    message(STATUS "Found PortAudio via PkgConfig/System Search.")
    # PORTAUDIO_INCLUDE_DIRS and PORTAUDIO_LIBRARIES should be set by pkg_check_modules
endif()


# -- nlohmann/json (Header-only, easy with FetchContent) --
set(JSON_FETCH_GIT_REPOSITORY https://github.com/nlohmann/json.git)
set(JSON_FETCH_GIT_TAG v3.11.3) # Use a specific tag
set(JSON_FOUND FALSE) # Flag to track if found

if(KWS_SDK_PREFER_SYSTEM_LIBS)
    # Try to find system-installed package via CMake config file
    # The nlohmann-json3-dev package on Ubuntu provides this
    find_package(nlohmann_json 3.2.0 CONFIG QUIET) # Check for version >= 3.2.0
    if(nlohmann_json_FOUND)
        message(STATUS "Found nlohmann/json via find_package (system). Version: ${nlohmann_json_VERSION}")
        set(JSON_FOUND TRUE)
        # The target nlohmann_json::nlohmann_json should be available
    endif()
endif()

# If not found via system search, use FetchContent
if(NOT JSON_FOUND)
    message(STATUS "nlohmann/json not found via find_package or system search disabled. Fetching from source.")
    FetchContent_Declare(
      nlohmann_json_fetch # Internal name for FetchContent
      GIT_REPOSITORY ${JSON_FETCH_GIT_REPOSITORY}
      GIT_TAG ${JSON_FETCH_GIT_TAG}
    )
    # This makes the `nlohmann_json::nlohmann_json` target available
    FetchContent_MakeAvailable(nlohmann_json_fetch)
    message(STATUS "nlohmann/json fetched/made available via FetchContent.")
    set(JSON_FOUND TRUE) # Mark as found via FetchContent
endif()

# Sanity check if we failed to find it either way
if(NOT JSON_FOUND)
     message(FATAL_ERROR "Failed to find or fetch nlohmann/json. Please install the 'nlohmann-json3-dev' package or check network connection.")
endif()


# -- FFTW (Example: Prefer system, fallback to FetchContent - FFTW is often system-installed) --
set(FFTW_FOUND FALSE) # Initialize flag

if(KWS_SDK_PREFER_SYSTEM_LIBS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        # Try pkg-config for the float version first (fftw3f)
        pkg_check_modules(FFTW3F fftw3f)
        if(FFTW3F_FOUND)
            message(STATUS "Found FFTW (float) via pkg-config (fftw3f).")
            set(FFTW_FOUND TRUE)
            set(FFTW_INCLUDE_DIRS ${FFTW3F_INCLUDE_DIRS})
            set(FFTW_LIBRARIES ${FFTW3F_LIBRARIES})
            set(FFTW_LINK_TARGET ${FFTW3F_LIBRARIES}) # Use library names directly from pkg-config
            message(STATUS "Using FFTW float libraries: ${FFTW_LINK_TARGET}")
        else()
            message(WARNING "pkg-config could not find 'fftw3f'. Trying 'fftw3'.")
            # If float not found, try pkg-config for the double version (fftw3)
            pkg_check_modules(FFTW3 fftw3)
            if(FFTW3_FOUND)
                message(STATUS "Found FFTW (double) via pkg-config (fftw3). Using double version.")
                set(FFTW_FOUND TRUE)
                set(FFTW_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
                set(FFTW_LIBRARIES ${FFTW3_LIBRARIES})
                set(FFTW_LINK_TARGET ${FFTW3_LIBRARIES})
                message(STATUS "Using FFTW double libraries: ${FFTW_LINK_TARGET}")
            endif()
        endif()
    else()
        message(WARNING "pkg-config tool not found. Cannot search for FFTW via pkg-config.")
    endif()
endif()

# Fallback / Error if not found via pkg-config
if(NOT FFTW_FOUND)
    # If system libs preferred but not found via pkg-config, error out
    if(KWS_SDK_PREFER_SYSTEM_LIBS)
         message(FATAL_ERROR "FFTW not found via pkg-config. Please ensure 'libfftw3-dev' is installed correctly and pkg-config can find 'fftw3f' or 'fftw3'. Check CMake output above for pkg-config errors.")
    else()
        # If system libs not preferred, original FetchContent logic could go here, but we keep it simple: error out.
         message(FATAL_ERROR "FFTW finding via pkg-config failed, and FetchContent is disabled or not implemented. Please install 'libfftw3-dev'.")
    endif()
endif()

# Ensure FFTW_LINK_TARGET is set if found (already done above)
if(FFTW_FOUND AND NOT FFTW_LINK_TARGET)
    message(WARNING "FFTW found via pkg-config, but FFTW_LINK_TARGET is not set correctly.")
    set(FFTW_LINK_TARGET ${FFTW_LIBRARIES})
endif()

# -- Eigen (Header-only, easy with FetchContent) --
# Note: GMM/HMM implementation might need Eigen

# Try finding system Eigen using pkg-config ONLY
set(EIGEN_FOUND FALSE)
if(KWS_SDK_PREFER_SYSTEM_LIBS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(EIGEN3 QUIET eigen3)
        if(EIGEN3_FOUND)
            message(STATUS "Found Eigen3 via pkg-config (eigen3). Version: ${EIGEN3_VERSION}")
            set(EIGEN_FOUND TRUE)
            include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
        else()
             message(WARNING "pkg-config could not find 'eigen3'.")
        endif()
    else()
        message(WARNING "pkg-config tool not found. Cannot search for Eigen3 via pkg-config.")
    endif()
else()
    message(WARNING "KWS_SDK_PREFER_SYSTEM_LIBS is OFF. System search for Eigen3 via pkg-config skipped.")
    # If system libs are not preferred, we are instructed not to fetch either.
    # So, Eigen will not be found unless KWS_SDK_PREFER_SYSTEM_LIBS is ON and pkg-config succeeds.
endif()

# If not found via system (pkg-config), ERROR OUT
if(NOT EIGEN_FOUND)
    message(FATAL_ERROR "Eigen3 not found via pkg-config. Please install the 'libeigen3-dev' package and ensure pkg-config can find 'eigen3'. FetchContent is disabled for Eigen3.")
endif()


# --- Add Subdirectories ---
add_subdirectory(src)

if(KWS_SDK_BUILD_DEMO)
    if(PORTAUDIO_FOUND) # Only build demo if PortAudio is available
        add_subdirectory(demo)
    else()
        message(WARNING "Skipping demo build because PortAudio dependency was not found or fetched correctly.")
    endif()
endif()

# --- Optional: Add Testing (e.g., using CTest) ---
# enable_testing()
# add_subdirectory(tests) # If you have a tests directory

# --- Optional: Installation ---
# Include installation rules defined in src/CMakeLists.txt if needed
# include(GNUInstallDirs)
# install(EXPORT kws_sdk_targets ...) # See src/CMakeLists.txt for example install rules 