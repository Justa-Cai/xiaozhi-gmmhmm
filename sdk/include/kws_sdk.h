#ifndef KWS_SDK_H
#define KWS_SDK_H

#include <vector>
#include <string>
#include <functional>
#include <map>
#include <deque>
#include <memory> // For std::unique_ptr
#include <chrono> // For time points
#include <limits> // For numeric_limits

// --- Configuration Structs ---
struct MfccParams {
    int num_cep = 13;
    double frame_len = 0.025;    // seconds
    double frame_stride = 0.01;  // seconds
    double preemph = 0.97;
    int sample_rate = 16000;     // Hz
    int nfft = 512; // FFT size (matches python_speech_features default)
    int num_mel_filters = 26; // Number of mel filters (matches python_speech_features default)
    double lowfreq = 0; // Lowest frequency for mel filters
    double highfreq = 0; // Highest frequency (0 or negative means sample_rate / 2)
    int ceplifter = 22; // Cepstral liftering coefficient
    bool appendEnergy = true; // Whether to append log energy
};

struct KwsConfig {
    MfccParams mfcc_params;      // MFCC parameters
    int sample_rate = 16000;     // Expected audio sample rate (Hz)
    int chunk_duration_ms = 100; // Duration of audio chunks processed (ms)
    double buffer_duration_s = 0.5; // Audio buffer length for feature extraction (s)
    double detection_threshold_offset = 300.0; // Log-likelihood diff threshold
    int smoothing_window_size = 8; // Number of detection frames to smooth over
    int min_detection_count = 3;   // Min counts in window to trigger
    double silence_threshold = 0.005; // RMS energy threshold for VAD
    std::string background_label = "background"; // Label for the background noise model
    double debounce_time_s = 1.0;   // Min time between consecutive triggers (s)
};

// --- Forward Declarations (Implementations in .cpp files) ---
// We forward declare internal implementation classes to avoid including
// their headers (and potentially dependencies like Eigen/JSON) in the public SDK header.
namespace kws {
    class GmmHmmModel; // Represents a loaded GMM-HMM model
    class FeatureScaler; // Represents the loaded scaler
    class KeywordSpotterImpl; // Forward declare the internal implementation class
    std::vector<std::vector<double>> extract_mfcc(
        const std::vector<float>& signal,
        const MfccParams& params
    );
    std::unique_ptr<FeatureScaler> load_scaler(const std::string& model_dir);
    std::map<std::string, std::unique_ptr<GmmHmmModel>> load_gmm_hmm_models(const std::string& model_dir);
}

// --- Detection Callback Type ---
// Function signature: void callback(const std::string& keyword, double detection_time_s);
// detection_time_s could be the time the keyword was confirmed.
using DetectionCallback = std::function<void(const std::string&, double)>;

// --- The Main SDK Class ---
class KeywordSpotter {
public:
    /**
     * @brief Constructor.
     * @param config Configuration parameters for the KWS system.
     */
    KeywordSpotter(const KwsConfig& config);

    /**
     * @brief Destructor.
     */
    ~KeywordSpotter();

    // Delete copy constructor and assignment operator to prevent accidental copying
    KeywordSpotter(const KeywordSpotter&) = delete;
    KeywordSpotter& operator=(const KeywordSpotter&) = delete;
    // Allow move constructor and assignment if needed (requires careful handling of resources)
    KeywordSpotter(KeywordSpotter&&) noexcept; 
    KeywordSpotter& operator=(KeywordSpotter&&) noexcept;

    /**
     * @brief Loads GMM-HMM models and the feature scaler from a specified directory.
     * Assumes models are exported in a C++ readable format (e.g., *.json) within the directory.
     * Must be called before processing audio.
     *
     * @param model_dir Path to the directory containing model files and scaler file.
     * @return true if loading was successful, false otherwise.
     */
    bool load_models(const std::string& model_dir);

    /**
     * @brief Processes a chunk of audio data.
     * Performs VAD, buffering, feature extraction, scoring, and detection logic.
     * Calls the registered detection callback if a keyword is detected.
     *
     * @param audio_chunk A vector containing PCM float audio samples for the chunk.
     *                    The number of samples should ideally correspond to `chunk_duration_ms`.
     */
    void process_audio_chunk(const std::vector<float>& audio_chunk);

    /**
     * @brief Registers a callback function to be invoked when a keyword is detected.
     * @param callback The function to call. Pass nullptr or empty function to unregister.
     */
    void set_detection_callback(DetectionCallback callback);

    /**
     * @brief Gets the current configuration.
     * @return const KwsConfig& A reference to the configuration struct.
     */
    const KwsConfig& get_config() const;

private:
    // --- Internal Helper Methods ---

    /** @brief Calculates RMS energy of a chunk for VAD. */
    double calculate_rms(const std::vector<float>& chunk) const;

    /** @brief Performs feature extraction, scaling, and scoring when buffer is ready. */
    void run_inference_on_buffer();

    /** @brief Updates the detection state based on current frame scores and applies smoothing/debouncing. */
    void update_detection_state(const std::map<std::string, double>& current_scores);

    // --- Member Variables (PIMPL approach) ---
    KwsConfig config_;
    // NO detection_callback_ here, it's in Impl now
    // NO models_loaded_ here, it's in Impl now

    // NO Audio Buffer members here

    // NO Models & Scaler members here

    // NO Detection State members here

    // Pointer to the implementation (PIMPL)
    std::unique_ptr<kws::KeywordSpotterImpl> impl_;
};

#endif // KWS_SDK_H 