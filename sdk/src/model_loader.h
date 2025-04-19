#ifndef KWS_MODEL_LOADER_H
#define KWS_MODEL_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <memory> // For std::unique_ptr

// Forward declare the base classes from the SDK header
namespace kws {
    // Keep forward declarations here as the public interface
    class GmmHmmModel;
    class FeatureScaler;
}

namespace kws {

/**
 * @brief Loads the feature scaler parameters from a file in the specified directory.
 * Expects a specific format (e.g., scaler.bin).
 *
 * @param model_dir The directory containing the scaler file.
 * @return std::unique_ptr<FeatureScaler> Pointer to the loaded scaler, or nullptr on failure.
 */
std::unique_ptr<FeatureScaler> load_scaler(const std::string& model_dir);


/**
 * @brief Loads all GMM-HMM models from files in the specified directory.
 * Expects files matching a pattern (e.g., gmmhmm_*.bin) and a specific format.
 *
 * @param model_dir The directory containing the model files.
 * @return std::map<std::string, std::unique_ptr<GmmHmmModel>> A map from model label (string) to the loaded model object. Empty map on failure or if no models are found.
 */
std::map<std::string, std::unique_ptr<GmmHmmModel>> load_gmm_hmm_models(const std::string& model_dir);

} // namespace kws

#endif // KWS_MODEL_LOADER_H 