#include "kws_sdk.h"
#include "model_loader.h" // Contains load_scaler, load_gmm_hmm_models declarations
#include "mfcc.h"         // Contains extract_mfcc declaration
#include "feature_scaler.h"
#include "gmm_hmm.h"      

#include <iostream>       // For std::cerr, std::cout
#include <cmath>          // For std::sqrt, std::log
#include <numeric>        // For std::accumulate
#include <algorithm>      // For std::max_element, std::find
#include <limits>         // For std::numeric_limits
#include <utility>        // For std::move
#include <map>            // Needed for update_detection_state
#include <vector>
#include <deque>
#include <chrono> 
#include <memory> // For unique_ptr

namespace kws {
    // --- PIMPL Implementation Struct --- 
    struct KeywordSpotterImpl {
        // Moved data members from KeywordSpotter header
        DetectionCallback detection_callback_ = nullptr;
        bool models_loaded_ = false;

        // Audio Buffer
        std::deque<float> audio_buffer_;
        size_t buffer_samples_max_ = 0;
        size_t chunk_samples_ = 0;
        size_t frame_len_samples_ = 0; 

        // Models & Scaler
        std::unique_ptr<kws::FeatureScaler> scaler_ = nullptr;
        std::map<std::string, std::unique_ptr<kws::GmmHmmModel>> models_;
        std::vector<std::string> keyword_labels_; 
        kws::GmmHmmModel* background_model_ = nullptr; 

        // Detection State
        std::deque<std::string> detection_queue_; 
        std::chrono::steady_clock::time_point last_detection_time_;

        // Store config copy if needed by implementation methods
        // KwsConfig config_; // We keep config_ in the main class for now

        // Constructor for Impl if needed (e.g., to calculate buffer sizes based on config)
        KeywordSpotterImpl(const KwsConfig& config, size_t smoothing_window_size, double debounce_time_s) { 
            buffer_samples_max_ = static_cast<size_t>(config.buffer_duration_s * config.sample_rate);
            chunk_samples_ = static_cast<size_t>((config.chunk_duration_ms / 1000.0) * config.sample_rate);
            frame_len_samples_ = static_cast<size_t>(config.mfcc_params.frame_len * config.sample_rate);
            detection_queue_.resize(smoothing_window_size, ""); 
            last_detection_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(static_cast<long>(debounce_time_s) + 1); 
        }

        // --- Implementation Helper Methods (Moved from KeywordSpotter) --- 
        // These methods now operate on the Impl members

        double calculate_rms(const std::vector<float>& chunk) const {
            if (chunk.empty()) {
                return 0.0;
            }
            double sum_sq = std::accumulate(chunk.begin(), chunk.end(), 0.0,
                                            [](double acc, float val) {
                                                return acc + static_cast<double>(val) * val;
                                            });
            return std::sqrt(sum_sq / chunk.size());
        }
        
        // Note: run_inference_on_buffer needs access to config_.mfcc_params
        // Pass config as argument or store a copy in Impl
        void run_inference_on_buffer(const MfccParams& mfcc_params) {
            if (audio_buffer_.size() < frame_len_samples_) { 
                return;
            }
            std::vector<float> signal_to_process(audio_buffer_.begin(), audio_buffer_.end());
            std::vector<std::vector<double>> features = kws::extract_mfcc(signal_to_process, mfcc_params);
            if (features.empty()) {
                return;
            }
            if (!scaler_) { /* Handle error */ return; }
            for (std::vector<double>& frame_features : features) {
                scaler_->transform(frame_features);
            }

            std::map<std::string, double> current_scores;
            for (auto const& [label, model_ptr] : models_) {
                try {
                    current_scores[label] = model_ptr->score(features); 
                } catch (const std::exception& e) {
                    std::cerr << "E: Scoring failed for model \"" << label << "\": " << e.what() << std::endl;
                    current_scores[label] = -std::numeric_limits<double>::infinity();
                }
            }
            // update_detection_state also needs config
            // We will call it from the main class method where config_ is available
            // This method should maybe just return the scores
            // update_detection_state(current_scores); 
        }
        
        // update_detection_state needs access to config and callback
        // Let's keep its logic in the main KeywordSpotter::update_detection_state method
        // void update_detection_state(const std::map<std::string, double>& current_scores) { ... }
    };

} // namespace kws

// --- KeywordSpotter Method Implementations --- 

// Constructor: Initialize PIMPL
KeywordSpotter::KeywordSpotter(const KwsConfig& config)
    : config_(config) // Store config directly
{ 
    // Create the implementation object, passing necessary config values
    impl_ = std::make_unique<kws::KeywordSpotterImpl>(config_, config_.smoothing_window_size, config_.debounce_time_s);

    std::cout << "KeywordSpotter initialized." << std::endl;
    std::cout << " - Sample Rate: " << config_.sample_rate << " Hz" << std::endl;
    std::cout << " - Chunk Samples: " << impl_->chunk_samples_ << " (" << config_.chunk_duration_ms << " ms)" << std::endl;
    std::cout << " - Buffer Samples: " << impl_->buffer_samples_max_ << " (" << config_.buffer_duration_s << " s)" << std::endl;
    std::cout << " - Frame Samples: " << impl_->frame_len_samples_ << " (" << config_.mfcc_params.frame_len * 1000 << " ms)" << std::endl;
}

// Destructor: MUST be defined here where KeywordSpotterImpl is complete
KeywordSpotter::~KeywordSpotter() = default; // Default implementation is fine now
// If std::cout message is desired:
// KeywordSpotter::~KeywordSpotter() {
//     std::cout << "KeywordSpotter destroyed." << std::endl;
// }

// Move constructor: Explicitly move the impl pointer
KeywordSpotter::KeywordSpotter(KeywordSpotter&& other) noexcept = default;
// If std::cout message is desired, or other logic:
// KeywordSpotter::KeywordSpotter(KeywordSpotter&& other) noexcept 
// : impl_(std::move(other.impl_)), config_(other.config_) {}

// Move assignment: Explicitly move the impl pointer
KeywordSpotter& KeywordSpotter::operator=(KeywordSpotter&& other) noexcept = default;
// If std::cout message is desired, or other logic:
// KeywordSpotter& KeywordSpotter::operator=(KeywordSpotter&& other) noexcept {
//     if (this != &other) {
//         impl_ = std::move(other.impl_);
//         config_ = other.config_;
//     }
//     return *this;
// }

// --- Public Method Implementations (Delegate to Impl or use Impl members) --- 

bool KeywordSpotter::load_models(const std::string& model_dir) {
    if (!impl_) return false; // Should not happen if constructed properly
    std::cout << "Loading models from directory: " << model_dir << std::endl;
    impl_->models_loaded_ = false; 
    impl_->models_.clear();
    impl_->keyword_labels_.clear();
    impl_->scaler_ = nullptr;
    impl_->background_model_ = nullptr;

    // Load Scaler
    impl_->scaler_ = kws::load_scaler(model_dir);
    if (!impl_->scaler_) {
        std::cerr << "Error: Failed to load scaler." << std::endl;
        return false;
    }
    std::cout << "Scaler loaded." << std::endl;

    // Load GMM-HMM Models
    impl_->models_ = kws::load_gmm_hmm_models(model_dir);
    if (impl_->models_.empty()) {
        std::cerr << "Error: Failed to load any models." << std::endl;
    }
    std::cout << impl_->models_.size() << " model(s) loaded initially." << std::endl;

    // Identify background model and cache keyword labels
    auto bg_it = impl_->models_.find(config_.background_label);
    if (bg_it != impl_->models_.end()) {
        impl_->background_model_ = bg_it->second.get(); 
        std::cout << "Background model (\"" << config_.background_label << "\") found." << std::endl;
    } else {
        std::cerr << "Warning: Background model (\"" << config_.background_label << "\") not found. Detection reliability may be reduced." << std::endl;
    }

    for (const auto& pair : impl_->models_) {
        if (pair.first != config_.background_label) {
            impl_->keyword_labels_.push_back(pair.first);
        }
    }

    if (impl_->keyword_labels_.empty() && impl_->background_model_) {
        std::cout << "Warning: Only background model loaded. No keywords will be detected." << std::endl;
    } else if (impl_->keyword_labels_.empty() && !impl_->background_model_){
         std::cout << "Warning: No keyword models or background model loaded." << std::endl;
    } else {
         std::cout << "Loaded keyword models: ";
         for(const auto& label : impl_->keyword_labels_) std::cout << label << " ";
         std::cout << std::endl;
    }

    impl_->models_loaded_ = (impl_->scaler_ != nullptr && !impl_->models_.empty()); 
    return impl_->models_loaded_;
}

// Helper function moved here as it needs impl_
double KeywordSpotter::calculate_rms(const std::vector<float>& chunk) const {
    // Direct implementation or call impl_->calculate_rms(chunk)
     if (chunk.empty()) {
        return 0.0;
    }
    double sum_sq = std::accumulate(chunk.begin(), chunk.end(), 0.0,
                                    [](double acc, float val) {
                                        return acc + static_cast<double>(val) * val;
                                    });
    return std::sqrt(sum_sq / chunk.size());
}

void KeywordSpotter::process_audio_chunk(const std::vector<float>& audio_chunk) {
    if (!impl_ || !impl_->models_loaded_) {
        return;
    }

    // --- VAD --- 
    double rms = calculate_rms(audio_chunk);
    if (rms < config_.silence_threshold) {
        return;
    }

    // --- Buffering (Access via impl_) --- 
    impl_->audio_buffer_.insert(impl_->audio_buffer_.end(), audio_chunk.begin(), audio_chunk.end());
    while (impl_->audio_buffer_.size() > impl_->buffer_samples_max_) {
        impl_->audio_buffer_.pop_front();
    }

    // --- Check if enough data for inference (Access via impl_) --- 
    if (impl_->audio_buffer_.size() >= impl_->frame_len_samples_) {
        run_inference_on_buffer(); // Call the private helper method
    }
}

// Private helper method
void KeywordSpotter::run_inference_on_buffer() {
    if (!impl_ || impl_->audio_buffer_.size() < impl_->frame_len_samples_) { 
        return;
    }

    std::vector<float> signal_to_process(impl_->audio_buffer_.begin(), impl_->audio_buffer_.end());

    // --- Feature Extraction (Uses config_ directly) --- 
    std::vector<std::vector<double>> features = kws::extract_mfcc(signal_to_process, config_.mfcc_params);

    if (features.empty()) {
        return;
    }

    // --- Feature Scaling (Uses impl_->scaler_) --- 
    if (!impl_->scaler_) { /* Handle error */ return; }
    for (std::vector<double>& frame_features : features) {
        impl_->scaler_->transform(frame_features);
    }

    // --- Scoring (Uses impl_->models_) --- 
    std::map<std::string, double> current_scores;
    for (auto const& [label, model_ptr] : impl_->models_) {
        try {
            current_scores[label] = model_ptr->score(features); 
        } catch (const std::exception& e) {
            std::cerr << "E: Scoring failed for model \"" << label << "\": " << e.what() << std::endl;
            current_scores[label] = -std::numeric_limits<double>::infinity();
        }
    }

    // --- Detection Logic (Call the private helper) --- 
    update_detection_state(current_scores);
}

// Private helper method
void KeywordSpotter::update_detection_state(const std::map<std::string, double>& current_scores) {
    if(!impl_) return;

    double max_keyword_score = -std::numeric_limits<double>::infinity();
    std::string current_best_keyword = ""; 

    // Use impl_->keyword_labels_
    for(const std::string& label : impl_->keyword_labels_) {
         auto it = current_scores.find(label);
         if (it != current_scores.end()) {
             if (it->second > max_keyword_score) {
                 max_keyword_score = it->second;
                 current_best_keyword = label;
             }
         }
    }

    // Use impl_->background_model_
    double background_score = -std::numeric_limits<double>::infinity();
    if (impl_->background_model_) {
        auto it = current_scores.find(config_.background_label);
        if (it != current_scores.end()) {
            background_score = it->second;
        }
    }

    // --- Thresholding (Uses config_) --- 
    std::string detected_keyword_this_frame = ""; 
    if (!current_best_keyword.empty() && 
        max_keyword_score > background_score + config_.detection_threshold_offset) 
    {
        detected_keyword_this_frame = current_best_keyword;
    } 

    // --- Smoothing (Uses impl_->detection_queue_) --- 
    if (!impl_->detection_queue_.empty()) {
        impl_->detection_queue_.pop_front();
        impl_->detection_queue_.push_back(detected_keyword_this_frame);
    } else {
        impl_->detection_queue_.resize(config_.smoothing_window_size, "");
        if (!impl_->detection_queue_.empty()) { // Check again after resize
             impl_->detection_queue_.back() = detected_keyword_this_frame;
        }
    }

    // Count occurrences in the smoothing window
    std::map<std::string, int> counts;
    int valid_detections = 0;
    for (const std::string& item : impl_->detection_queue_) {
        if (!item.empty()) { 
            counts[item]++;
            valid_detections++;
        }
    }

    std::string final_detection = ""; 
    for (const auto& pair : counts) {
        // Use config_ for min_detection_count
        if (pair.second >= config_.min_detection_count) {
            final_detection = pair.first;
            break;
        }
    }

    // --- Debouncing & Triggering Callback (Uses impl_ members and config_) --- 
    if (!final_detection.empty()) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_since_last = now - impl_->last_detection_time_;

        if (time_since_last.count() >= config_.debounce_time_s) {
             // Use impl_->detection_callback_
             if (impl_->detection_callback_) {
                 try {
                     double detection_time = std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
                     impl_->detection_callback_(final_detection, detection_time); 
                 } catch (const std::exception& e) {
                     std::cerr << "Error invoking detection callback: " << e.what() << std::endl;
                 }
             } else {
                 std::cout << "\n>>> Detected Keyword: [ " << final_detection << " ] <<<" << std::endl;
             }
            
             impl_->last_detection_time_ = now; 
             // Use config_ for smoothing_window_size
             impl_->detection_queue_.assign(config_.smoothing_window_size, ""); 
        }
    }
}

void KeywordSpotter::set_detection_callback(DetectionCallback callback) {
    if (impl_) {
        impl_->detection_callback_ = std::move(callback);
    }
}

const KwsConfig& KeywordSpotter::get_config() const {
    // config_ is a direct member, return it
    return config_;
}
 