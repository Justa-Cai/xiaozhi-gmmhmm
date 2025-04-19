#ifndef KWS_FEATURE_SCALER_H
#define KWS_FEATURE_SCALER_H

#include <vector>
#include <string>
#include <iostream> // For cerr
#include <stdexcept> // For runtime_error
#include <cmath> // For isnan, isinf

namespace kws {

/**
 * @brief Concrete implementation of the feature scaler.
 * Stores mean and inverse standard deviation for efficient scaling.
 */
class FeatureScaler {
public:
    FeatureScaler(std::vector<float> mean, std::vector<float> inv_std_dev)
        : mean_(std::move(mean)), inv_std_dev_(std::move(inv_std_dev)), dim_(mean_.size()) {}

    /**
     * @brief Applies scaling to a feature vector in-place.
     * scaled = (feature - mean) * inv_std_dev
     *
     * @param features The feature vector (std::vector<double>) to scale.
     *                 Note: Internal calculations use float, consistent with Python export.
     *                 The input is double to match the output of the planned MFCC function.
     */
    void transform(std::vector<double>& features) const {
        if (features.size() != dim_) {
            std::cerr << "Error: Feature dimension mismatch in scaler. Expected " << dim_
                      << ", got " << features.size() << ". Skipping scaling." << std::endl;
            // throw std::runtime_error("Scaler dimension mismatch");
            return;
        }
        for (size_t i = 0; i < dim_; ++i) {
            // Cast input double to float for calculation, store back as double
            float scaled_val = (static_cast<float>(features[i]) - mean_[i]) * inv_std_dev_[i];
            // Check for NaN/Inf after scaling (can happen if inv_std_dev is inf/nan or input is inf)
            if (std::isnan(scaled_val) || std::isinf(scaled_val)) {
                 // Handle error: maybe replace with 0 or throw?
                 // Replacing with 0 for now, similar to potential HMM training behavior
                 std::cerr << "Warning: NaN/Inf encountered during feature scaling at index " << i << ". Replacing with 0." << std::endl;
                 features[i] = 0.0;
            } else {
                features[i] = static_cast<double>(scaled_val);
            }
        }
    }

    size_t get_dimension() const { return dim_; }

private:
    std::vector<float> mean_;
    std::vector<float> inv_std_dev_;
    size_t dim_;
};

} // namespace kws

#endif // KWS_FEATURE_SCALER_H 