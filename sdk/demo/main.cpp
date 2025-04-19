#include <kws_sdk.h> // Include our SDK header
#include <portaudio.h> // Include PortAudio header
#include <iostream>
#include <vector>
#include <csignal> // For signal handling (Ctrl+C)
#include <atomic>  // For thread-safe flag
#include <chrono>
#include <thread> // For sleep
#include <iomanip> // For std::put_time with C++11 (alternative to strftime)

// --- Configuration ---
const std::string MODEL_DIR = "./models"; // Relative path from build/sdk/demo/ to models/
                                           // Adjust this path as needed based on your build structure
                                           // Or use an absolute path

// --- Global Variables for Demo --- (Declared here)
std::atomic_flag keep_running = ATOMIC_FLAG_INIT;
std::chrono::steady_clock::time_point start_time;
std::chrono::steady_clock::time_point last_keyword_detected_time;

// --- PortAudio Callback Data ---
struct PaCallbackData {
    KeywordSpotter* spotter = nullptr;
    std::vector<float> chunk_buffer; // Pre-allocate buffer for audio chunks
};

// --- Global flag for Ctrl+C ---
std::atomic<bool> g_interrupted(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nCtrl+C detected, stopping..." << std::endl;
        g_interrupted = true;
    }
}

// --- PortAudio Callback Function ---
static int pa_callback(const void *inputBuffer, void *outputBuffer,
                       unsigned long framesPerBuffer,
                       const PaStreamCallbackTimeInfo* timeInfo,
                       PaStreamCallbackFlags statusFlags,
                       void *userData) {
    PaCallbackData *data = (PaCallbackData*)userData;
    const float *rptr = (const float*)inputBuffer; // Input is float as requested

    (void) outputBuffer; // Prevent unused variable warning
    (void) timeInfo;
    (void) statusFlags;

    if (inputBuffer == nullptr) {
        std::cerr << "Warning: PortAudio input buffer is null." << std::endl;
        return paContinue;
    }

    // Ensure the chunk buffer is the correct size
    if (data->chunk_buffer.size() != framesPerBuffer) {
        data->chunk_buffer.resize(framesPerBuffer);
    }

    // Copy input data to our buffer
    std::copy(rptr, rptr + framesPerBuffer, data->chunk_buffer.begin());

    // Process the chunk using the KeywordSpotter SDK
    if (data->spotter) {
        try {
            data->spotter->process_audio_chunk(data->chunk_buffer);
        } catch (const std::exception& e) {
            std::cerr << "\nError processing audio chunk in SDK: " << e.what() << std::endl;
            // Decide if the error is fatal for the stream
            // return paAbort;
        }
    }

    // Check interruption flag
    return g_interrupted ? paComplete : paContinue;
}

// --- Keyword Detection Callback Function ---
void on_keyword_detected(const std::string& keyword, double score) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    // Clear the current line
    std::cout << "\r" << std::string(80, ' ') << "\r"; // Clear line
    std::cout << "[INFO] [" << std::setw(4) << elapsed << "s] Keyword '" << keyword << "' detected! Score: " << score << std::endl;
    last_keyword_detected_time = now; // Update last detection time
}

// Callback function for processing audio (provides feedback)
void on_audio_processing(size_t chunk_count) {
    auto now = std::chrono::steady_clock::now();
    // Only print dot if no keyword detected recently
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_keyword_detected_time).count() > 2) {
        std::cout << "."; // Print dot to show activity continues
        std::cout.flush();
    }
}

int main() {
    std::cout << "KWS SDK Demo Application" << std::endl;
    std::cout << "------------------------" << std::endl;

    // --- Setup Signal Handler ---
    signal(SIGINT, signal_handler);

    // --- Configure KWS ---
    KwsConfig kws_config;
    // Use default config values initially, same as Python script if possible
    // Print the config being used
    std::cout << "Using KWS Configuration:" << std::endl;
    std::cout << "  Sample Rate: " << kws_config.sample_rate << " Hz" << std::endl;
    std::cout << "  Chunk Duration: " << kws_config.chunk_duration_ms << " ms" << std::endl;
    std::cout << "  Buffer Duration: " << kws_config.buffer_duration_s << " s" << std::endl;
    std::cout << "  Threshold Offset: " << kws_config.detection_threshold_offset << std::endl;
    std::cout << "  Smoothing Window: " << kws_config.smoothing_window_size << std::endl;
    std::cout << "  Min Detection Count: " << kws_config.min_detection_count << std::endl;
    std::cout << "  Silence Threshold: " << kws_config.silence_threshold << std::endl;
    std::cout << "  Debounce Time: " << kws_config.debounce_time_s << " s" << std::endl;
    std::cout << "  Background Label: " << kws_config.background_label << std::endl;
    std::cout << "  MFCC Num Cep: " << kws_config.mfcc_params.num_cep << std::endl;
    // ... print other MFCC params ...

    // --- Initialize Keyword Spotter ---
    KeywordSpotter spotter(kws_config);

    // --- Load Models ---
    std::cout << "\nAttempting to load models from: " << MODEL_DIR << std::endl;
    if (!spotter.load_models(MODEL_DIR)) {
        std::cerr << "FATAL: Failed to load models. Please ensure models exist in '"
                  << MODEL_DIR << "' and are in the correct C++-readable format (e.g., *.json)." << std::endl;
        std::cerr << "Ensure the model directory path is correct relative to the executable location." << std::endl;
        return 1;
    }
    std::cout << "Models loaded successfully." << std::endl;

    // --- Set Detection Callback ---
    spotter.set_detection_callback(on_keyword_detected);
    std::cout << "Detection callback registered." << std::endl;

    // --- Initialize PortAudio ---
    PaError err = paNoError;
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_Initialize returned " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }
    std::cout << "PortAudio initialized." << std::endl;

    // --- Prepare PortAudio Callback Data ---
    PaCallbackData pa_data;
    pa_data.spotter = &spotter;
    // Calculate chunk size in frames for PortAudio
    unsigned long frames_per_chunk = static_cast<unsigned long>((kws_config.chunk_duration_ms / 1000.0) * kws_config.sample_rate);
    pa_data.chunk_buffer.resize(frames_per_chunk); // Pre-allocate
    std::cout << "PortAudio Frames Per Buffer (Chunk): " << frames_per_chunk << std::endl;


    // --- Open PortAudio Stream ---
    PaStream *stream = nullptr;
    PaStreamParameters inputParameters;

    inputParameters.device = Pa_GetDefaultInputDevice(); // Use default input device
    if (inputParameters.device == paNoDevice) {
        std::cerr << "PortAudio error: No default input device found." << std::endl;
        Pa_Terminate();
        return 1;
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParameters.device);
    std::cout << "Using PortAudio Input Device: " << deviceInfo->name << std::endl;


    inputParameters.channelCount = 1; // Mono input
    inputParameters.sampleFormat = paFloat32; // Use float32 as required by SDK/Python script
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(
              &stream,
              &inputParameters,
              nullptr, // No output
              kws_config.sample_rate,
              frames_per_chunk, // Request frames per buffer = chunk size
              paClipOff, // Don't clip samples
              pa_callback, // Your callback function
              &pa_data);   // User data passed to callback

    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_OpenStream returned " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }
    std::cout << "PortAudio stream opened." << std::endl;

    // --- Start PortAudio Stream ---
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_StartStream returned " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    std::cout << "\nMicrophone stream started. Listening... Press Ctrl+C to stop." << std::endl;

    // --- Main Loop (Wait for interrupt) ---
    while (Pa_IsStreamActive(stream) && !g_interrupted) {
        // Print a dot occasionally to show the main thread is alive
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // --- Stop PortAudio Stream ---
    std::cout << "\nStopping PortAudio stream..." << std::endl;
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_StopStream returned " << Pa_GetErrorText(err) << std::endl;
        // Continue cleanup regardless
    }

    // --- Close PortAudio Stream ---
    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_CloseStream returned " << Pa_GetErrorText(err) << std::endl;
    }

    // --- Terminate PortAudio ---
    err = Pa_Terminate();
    if (err != paNoError) {
        std::cerr << "PortAudio error: Pa_Terminate returned " << Pa_GetErrorText(err) << std::endl;
    }
    std::cout << "PortAudio terminated." << std::endl;

    std::cout << "KWS SDK Demo finished." << std::endl;
    return 0;
} 