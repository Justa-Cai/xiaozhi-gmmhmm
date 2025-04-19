#include "model_loader.h"
#include "kws_sdk.h" // Include the main SDK header for definitions
#include <iostream> // For error messages
#include <stdexcept> // For exceptions if needed
#include <filesystem> // For directory iteration (C++17)
#include <fstream> // For file reading
#include <cmath> // For std::log
#include <numeric> // For std::accumulate
#include <limits> // For std::numeric_limits

// If using Eigen for matrix/vector operations within GmmHmmModel:
// #include <Eigen/Core>

// Define constants used for filenames
const std::string SCALER_BIN_FILENAME = "scaler.bin";
const std::string MODEL_BIN_FILENAME_PREFIX = "gmmhmm_";
const std::string MODEL_BIN_FILENAME_SUFFIX = ".bin";

// Include headers for the concrete classes used here
#include "feature_scaler.h"
#include "gmm_hmm.h" 

namespace kws {

// --- Concrete Class Definitions REMOVED --- 
// Definitions are now in feature_scaler.h and gmm_hmm.h

// --- Implementation of Loading Functions --- 

/**
 * @brief Helper function to read a fixed number of bytes.
 */
template<typename T>
bool read_value(std::ifstream& ifs, T& value) {
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template<typename T>
bool read_vector(std::ifstream& ifs, std::vector<T>& vec, size_t count) {
    vec.resize(count);
    if (count == 0) return true; // Reading 0 bytes is success
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * count));
}

std::unique_ptr<FeatureScaler> load_scaler(const std::string& model_dir) {
    std::filesystem::path scaler_path = std::filesystem::path(model_dir) / SCALER_BIN_FILENAME;
    std::cout << "Attempting to load binary scaler from: " << scaler_path << std::endl;

    std::ifstream ifs(scaler_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Cannot open scaler file: " << scaler_path << std::endl;
        return nullptr;
    }

    try {
        int feature_dim = 0;
        if (!read_value(ifs, feature_dim) || feature_dim <= 0) {
            std::cerr << "Error: Failed to read valid feature dimension from scaler file." << std::endl;
            return nullptr;
        }

        std::vector<float> mean;
        if (!read_vector(ifs, mean, feature_dim)) {
            std::cerr << "Error: Failed to read mean vector from scaler file." << std::endl;
            return nullptr;
        }

        std::vector<float> inv_std_dev;
        if (!read_vector(ifs, inv_std_dev, feature_dim)) {
            std::cerr << "Error: Failed to read inverse std dev vector from scaler file." << std::endl;
            return nullptr;
        }

        // Check if we reached EOF unexpectedly
        ifs.peek(); 
        if (!ifs.eof()) {
            std::streampos current = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            std::streampos end = ifs.tellg();
            ifs.seekg(current); // Restore position
            std::cerr << "Warning: Extra data found at the end of scaler file: " 
                      << scaler_path << " (" << (end-current) << " bytes remaining)." << std::endl;
        }

        std::cout << "Scaler loaded successfully. Feature dim: " << feature_dim << std::endl;
        return std::make_unique<FeatureScaler>(std::move(mean), std::move(inv_std_dev));

    } catch (const std::ios_base::failure& e) {
        std::cerr << "Error during file I/O for scaler " << scaler_path << ": " << e.what() << " (I/O state: " << ifs.rdstate() << ")" << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Error reading or parsing scaler file " << scaler_path << ": " << e.what() << std::endl;
        return nullptr;
    }
}

std::map<std::string, std::unique_ptr<GmmHmmModel>> load_gmm_hmm_models(const std::string& model_dir) {
    std::map<std::string, std::unique_ptr<GmmHmmModel>> models;
    const std::string model_prefix = MODEL_BIN_FILENAME_PREFIX;
    const std::string model_suffix = MODEL_BIN_FILENAME_SUFFIX;

    std::cout << "Scanning for binary models in: " << model_dir << std::endl;

    if (!std::filesystem::is_directory(model_dir)) {
        std::cerr << "Error: Model directory not found or is not a directory: " << model_dir << std::endl;
        return models; 
    }

    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.size() > model_prefix.size() + model_suffix.size() &&
                filename.compare(0, model_prefix.size(), model_prefix) == 0 &&
                filename.compare(filename.size() - model_suffix.size(), model_suffix.size(), model_suffix) == 0)
            {
                std::string label_from_filename = filename.substr(model_prefix.size(), filename.size() - model_prefix.size() - model_suffix.size());
                std::cout << "Found potential model file: " << filename << " for label: " << label_from_filename << std::endl;

                std::ifstream ifs(entry.path(), std::ios::binary);
                if (!ifs) {
                    std::cerr << "Error: Cannot open model file: " << entry.path() << std::endl;
                    continue;
                }
                // Enable exceptions for the stream for easier error handling
                ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

                try {
                    // 1. Read label
                    int label_len = 0;
                    read_value(ifs, label_len);
                    if (label_len <= 0 || label_len > 256) { 
                        std::cerr << "Error: Invalid label length (" << label_len << ") read from " << filename << std::endl; continue;
                    }
                    std::vector<char> label_buffer(label_len);
                    ifs.read(label_buffer.data(), label_len);
                    std::string label(label_buffer.begin(), label_buffer.end());
                    if (label != label_from_filename) {
                         std::cerr << "Warning: Label mismatch in file '" << filename << "'. Expected '"
                                   << label_from_filename << "', got '" << label << "'. Using label from file." << std::endl;
                    }

                    // 2. Read HMM parameters
                    int num_states = 0, num_features = 0;
                    read_value(ifs, num_states);
                    read_value(ifs, num_features);
                    if (num_states <= 0 || num_features <= 0) {
                        std::cerr << "Error: Invalid num_states (" << num_states << ") or num_features (" << num_features << ") read from " << filename << std::endl; continue;
                    }

                    // 3. Read log_start_prob
                    std::vector<float> log_start_prob;
                    read_vector(ifs, log_start_prob, num_states);
                    
                    // 4. Read log_trans_mat
                    std::vector<float> log_trans_mat;
                    read_vector(ifs, log_trans_mat, num_states * num_states);
                    
                    // 5. Read GMM parameters per state
                    std::vector<GmmState> states(num_states);
                    for (int i = 0; i < num_states; ++i) {
                        states[i].num_features = num_features; 
                        // a. Read num_mix
                        read_value(ifs, states[i].num_mix);
                        if (states[i].num_mix <= 0) {
                            std::cerr << "Error: Invalid num_mix (" << states[i].num_mix << ") for state " << i << " in " << filename << std::endl; goto next_file; 
                        }
                        // b. Read log_weights
                        read_vector(ifs, states[i].log_weights, states[i].num_mix);
                        // c. Read means
                        read_vector(ifs, states[i].means, states[i].num_mix * num_features);
                        // d. Read inv_variances
                        read_vector(ifs, states[i].inv_variances, states[i].num_mix * num_features);
                        // e. Read log_gconsts
                        read_vector(ifs, states[i].log_gconsts, states[i].num_mix);
                    }

                    // Check for extra data
                    ifs.peek();
                    if (!ifs.eof()) {
                       std::streampos current = ifs.tellg();
                       ifs.exceptions(std::ifstream::goodbit); // Temporarily disable exceptions for seekg/tellg
                       ifs.seekg(0, std::ios::end);
                       std::streampos end = ifs.tellg();
                       ifs.seekg(current); 
                       ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit); // Re-enable exceptions
                       std::cerr << "Warning: Extra data found at the end of model file: " 
                                 << entry.path() << " (" << (end-current) << " bytes remaining)." << std::endl;
                    }

                    // Create and store the model
                    auto model_ptr = std::make_unique<GmmHmmModel>(
                        label,
                        num_states,
                        num_features,
                        std::move(log_start_prob),
                        std::move(log_trans_mat),
                        std::move(states));

                    models[label] = std::move(model_ptr);
                    std::cout << "Successfully loaded model for '" << label << "'" << std::endl;

                } catch (const std::ios_base::failure& e) {
                     std::cerr << "Error during file I/O for model " << entry.path() << ": " << e.what() << " (I/O state: " << ifs.rdstate() << ")" << std::endl;
                     // Continue to next file
                } catch (const std::exception& e) {
                    std::cerr << "Error loading or parsing model file " << entry.path() << ": " << e.what() << std::endl;
                    // Continue trying to load other models
                }
            next_file:; // Label for goto
            }
        }
    }

    if (models.empty()) {
         std::cerr << "Warning: No models were successfully loaded from " << model_dir << std::endl;
    }

    return models;
}

} // namespace kws
