# CMakeLists.txt for kws_demo application

# Define the executable target
add_executable(kws_demo main.cpp)

# Link the demo against the kws_sdk library
# kws_sdk target is defined in src/CMakeLists.txt and added via add_subdirectory in the top-level CMakeLists.txt
target_link_libraries(kws_demo PRIVATE kws_sdk)

# Link against PortAudio
# We need to handle both finding it via system (pkg-config) and via FetchContent
if(PORTAUDIO_FOUND)
    # Include directories might be needed explicitly even when linking target
    if(PORTAUDIO_INCLUDE_DIRS)
        target_include_directories(kws_demo PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    endif()

    # Link the library/target
    target_link_libraries(kws_demo PRIVATE ${PORTAUDIO_LIBRARIES}) # This variable holds the target name or library path
    message(STATUS "Linking demo against PortAudio: ${PORTAUDIO_LIBRARIES}")
else()
    # This case should ideally not be reached if the top-level CMakeLists.txt handles PORTAUDIO_FOUND correctly
    message(WARNING "PortAudio dependency information not found. Demo linking might fail.")
    # You might need to manually specify paths or library names here as a fallback, but it's fragile.
endif()

# Link against other libraries needed by the demo itself (if any)
# For example, if the demo directly uses threads (beyond PortAudio's usage)
# find_package(Threads REQUIRED)
# target_link_libraries(kws_demo PRIVATE Threads::Threads)

# Set C++ standard (should match SDK)
set_target_properties(kws_demo PROPERTIES CXX_STANDARD 17)
set_target_properties(kws_demo PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(kws_demo PROPERTIES CXX_EXTENSIONS NO)


# --- Optional: Installation ---
# Install the demo executable if needed
# include(GNUInstallDirs)
# install(TARGETS kws_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) 