#ifndef KWS_GMM_HMM_H
#define KWS_GMM_HMM_H

#include <vector>
#include <string>
#include <limits>   // For std::numeric_limits
#include <iostream> // For warnings/errors
#include <cmath>    // For std::log, std::exp
#include <numeric>  // For std::accumulate
#include <stdexcept> // For std::runtime_error
#include <map>      // Needed if score result uses map
#include <memory>   // For unique_ptr if used internally

// This header defines the *interface* or declaration of the GMM-HMM model class.
// The actual implementation (scoring logic) goes into gmm_hmm.cpp.
// You might need to include Eigen or other linear algebra headers here if they
// are part of the public interface (e.g., if score() takes Eigen types directly),
// but it's generally better to keep dependencies hidden in the .cpp file if possible.

namespace kws {

/**
 * @brief Represents the parameters of a single Gaussian component within a GMM.
 * (Definition moved here from model_loader.cpp)
 */
struct GaussianComponent {
    // No members needed here as params are flattened in GmmState
};

/**
 * @brief Represents a state in the GMM-HMM, containing its GMM parameters.
 * (Definition moved here from model_loader.cpp)
 */
struct GmmState {
    int num_mix = 0;
    int num_features = 0;
    std::vector<float> log_weights;   // Size = num_mix
    std::vector<float> means;         // Size = num_mix * num_features (row-major)
    std::vector<float> inv_variances; // Size = num_mix * num_features (row-major)
    std::vector<float> log_gconsts;   // Size = num_mix

    /**
     * @brief Calculate the log likelihood of a feature vector for this state's GMM.
     * Uses logsumexp for numerical stability.
     */
    float calculate_gmm_log_likelihood(const std::vector<double>& feature) const {
        if (feature.size() != num_features) {
             std::cerr << "Error: Feature dimension mismatch in GMM likelihood calculation. Expected "
                       << num_features << ", got " << feature.size() << ". Returning -inf." << std::endl;
            return -std::numeric_limits<float>::infinity();
        }

        std::vector<float> log_likelihoods(num_mix);
        for (int k = 0; k < num_mix; ++k) {
            float log_mahalanobis_term = 0.0f;
            size_t base_idx = k * num_features;
            for (int d = 0; d < num_features; ++d) {
                // Check for NaN/Inf in input features
                if (std::isnan(feature[d]) || std::isinf(feature[d])) {
                     std::cerr << "Warning: NaN/Inf found in feature vector at dim " << d << ". Cannot calculate likelihood." << std::endl;
                     return -std::numeric_limits<float>::infinity(); // Cannot proceed
                }
                float diff = static_cast<float>(feature[d]) - means[base_idx + d];
                // Check for NaN/Inf in parameters (should not happen if loaded correctly)
                if (std::isnan(means[base_idx + d]) || std::isinf(means[base_idx + d]) ||
                    std::isnan(inv_variances[base_idx + d]) || std::isinf(inv_variances[base_idx + d])) {
                    std::cerr << "Warning: NaN/Inf found in GMM parameters (mean/inv_var) for mix " << k << ", dim " << d << "." << std::endl;
                    return -std::numeric_limits<float>::infinity(); // Cannot proceed
                }
                log_mahalanobis_term += diff * diff * inv_variances[base_idx + d];
            }

            // Check term before 0.5 multiplication
            if (std::isnan(log_mahalanobis_term) || std::isinf(log_mahalanobis_term)) {
                 std::cerr << "Warning: NaN/Inf in Mahalanobis term for mix " << k << "." << std::endl;
                 log_likelihoods[k] = -std::numeric_limits<float>::infinity();
                 continue; // Try next mixture component
            }

            // log N(O | mean_k, var_k) = log_gconsts[k] - 0.5 * sum(...)
            float log_gaussian_prob = log_gconsts[k] - 0.5f * log_mahalanobis_term;

            // Check parameters and result
             if (std::isnan(log_gconsts[k]) || std::isinf(log_gconsts[k]) ||
                 std::isnan(log_weights[k]) || std::isinf(log_weights[k]) ||
                 std::isnan(log_gaussian_prob) || std::isinf(log_gaussian_prob)) {
                  std::cerr << "Warning: NaN/Inf in GMM parameters (gconst/weight) or intermediate gaussian prob for mix " << k << "." << std::endl;
                  log_likelihoods[k] = -std::numeric_limits<float>::infinity();
                  continue; // Try next mixture component
             }

            log_likelihoods[k] = log_weights[k] + log_gaussian_prob;
        }

        // --- logsumexp --- 
        if (log_likelihoods.empty()) {
            return -std::numeric_limits<float>::infinity();
        }
        float max_val = -std::numeric_limits<float>::infinity();
        bool has_finite = false;
        for (float ll : log_likelihoods) {
            if (ll > max_val) {
                max_val = ll;
            }
             if (ll != -std::numeric_limits<float>::infinity()) {
                 has_finite = true;
             }
        }
        if (!has_finite) { // If all are -inf
             return -std::numeric_limits<float>::infinity();
        }

        float sum_exp = 0.0f;
        for (float val : log_likelihoods) {
             if (val != -std::numeric_limits<float>::infinity()) { 
                 sum_exp += std::exp(val - max_val);
             }
        }
        // Check sum_exp before log
        if (sum_exp <= 0.0f || std::isnan(sum_exp) || std::isinf(sum_exp)) {
             std::cerr << "Warning: Invalid sum_exp (" << sum_exp << ") in logsumexp. Returning max_val." << std::endl;
             return max_val; // Return the max log-likelihood found as a fallback
        }

        return max_val + std::log(sum_exp);
    }
};


/**
 * @brief Concrete implementation of a GMM-HMM model.
 * (Definition moved here from model_loader.cpp)
 */
class GmmHmmModel {
public:
    GmmHmmModel(std::string label, int num_states, int num_features,
                std::vector<float> log_start_prob, std::vector<float> log_trans_mat,
                std::vector<GmmState> states)
        : label_(std::move(label)),
          num_states_(num_states),
          num_features_(num_features),
          log_start_prob_(std::move(log_start_prob)),
          log_trans_mat_(std::move(log_trans_mat)),
          states_(std::move(states))
    {}

    /**
     * @brief Calculates the log-likelihood of a sequence of feature vectors using this model.
     * Uses the Forward Algorithm.
     */
    double score(const std::vector<std::vector<double>>& features) const {
        size_t T = features.size(); // Number of frames (time steps)
        if (T == 0) {
            return -std::numeric_limits<double>::infinity();
        }
        // Check feature dimension of the first frame
        if (!features.empty() && features[0].size() != num_features_) {
             std::cerr << "Error: Feature dimension mismatch in HMM scoring. Expected " << num_features_
                       << ", got " << features[0].size() << ". Returning -inf." << std::endl;
            return -std::numeric_limits<double>::infinity();
        }

        // alpha[t][j] = log probability of being in state j after observing O_0, ..., O_t
        std::vector<std::vector<float>> log_alpha(T, std::vector<float>(num_states_, -std::numeric_limits<float>::infinity()));

        // 1. Initialization (t=0)
        for (int j = 0; j < num_states_; ++j) {
            float log_emission_prob = states_[j].calculate_gmm_log_likelihood(features[0]);
            if (log_start_prob_[j] != -std::numeric_limits<float>::infinity() && log_emission_prob != -std::numeric_limits<float>::infinity()) {
                 log_alpha[0][j] = log_start_prob_[j] + log_emission_prob;
            }
        }

        // 2. Recursion (t=1 to T-1)
        std::vector<float> log_probs_prev_sum(num_states_); // Temporary storage for logsumexp input
        for (size_t t = 1; t < T; ++t) {
            // Check feature dimension for this frame
            if (features[t].size() != num_features_) {
                 std::cerr << "Error: Feature dimension mismatch in HMM scoring at frame " << t << ". Expected " << num_features_
                       << ", got " << features[t].size() << ". Returning -inf." << std::endl;
                 return -std::numeric_limits<double>::infinity();
            }
            for (int j = 0; j < num_states_; ++j) {
                for (int i = 0; i < num_states_; ++i) {
                     float log_trans = get_log_trans_prob(i, j);
                     if (log_alpha[t - 1][i] != -std::numeric_limits<float>::infinity() && log_trans != -std::numeric_limits<float>::infinity()) {
                        log_probs_prev_sum[i] = log_alpha[t - 1][i] + log_trans;
                     } else {
                        log_probs_prev_sum[i] = -std::numeric_limits<float>::infinity();
                     }
                }
                float log_sum_term = log_sum_exp(log_probs_prev_sum);

                if (log_sum_term != -std::numeric_limits<float>::infinity()) {
                    float log_emission_prob = states_[j].calculate_gmm_log_likelihood(features[t]);
                    if (log_emission_prob != -std::numeric_limits<float>::infinity()) {
                        log_alpha[t][j] = log_emission_prob + log_sum_term;
                    }
                }
            }
        }

        // 3. Termination
        // Total log likelihood = logsumexp(log_alpha[T-1][i]) over all i
        double final_log_likelihood = log_sum_exp(log_alpha[T - 1]);

        return final_log_likelihood;
    }

    const std::string& get_label() const { return label_; }
    int get_num_states() const { return num_states_; }
    int get_num_features() const { return num_features_; }

private:
    friend class KeywordSpotter; // Allow KeywordSpotter to access private members if needed
    friend std::map<std::string, std::unique_ptr<GmmHmmModel>> load_gmm_hmm_models(const std::string& model_dir);

    std::string label_;
    int num_states_;
    int num_features_;
    std::vector<float> log_start_prob_; // Size = num_states
    std::vector<float> log_trans_mat_;  // Size = num_states * num_states (row-major)
    std::vector<GmmState> states_;      // Size = num_states

    // Helper to get log transition probability
    float get_log_trans_prob(int from_state, int to_state) const {
        if (from_state < 0 || from_state >= num_states_ || to_state < 0 || to_state >= num_states_) {
            return -std::numeric_limits<float>::infinity(); // Invalid transition
        }
        // Bounds check for flattened matrix access
        size_t index = static_cast<size_t>(from_state) * num_states_ + to_state;
        if (index >= log_trans_mat_.size()) {
             std::cerr << "Error: Transition matrix access out of bounds (" << from_state << " -> " << to_state << ")." << std::endl;
             return -std::numeric_limits<float>::infinity();
        }
        return log_trans_mat_[index];
    }

    // --- LogSumExp (utility) ---
    static float log_sum_exp(const std::vector<float>& log_probs) {
        if (log_probs.empty()) {
            return -std::numeric_limits<float>::infinity();
        }
        float max_val = -std::numeric_limits<float>::infinity();
        bool has_finite = false;
        for(float p : log_probs) {
             if (p != -std::numeric_limits<float>::infinity()) {
                has_finite = true;
                if (p > max_val) {
                     max_val = p;
                }
             }
        }
        if (!has_finite) {
             return -std::numeric_limits<float>::infinity();
        }

        float sum_exp = 0.0f;
        for (float p : log_probs) {
            if (p != -std::numeric_limits<float>::infinity()) {
                 sum_exp += std::exp(p - max_val);
            }
        }
         // Handle potential numerical issues with sum_exp
        if (sum_exp <= 0.0f || std::isnan(sum_exp) || std::isinf(sum_exp)) {
             // Log near zero or error, return max_val as best estimate
             // std::cerr << "Warning: sum_exp (" << sum_exp << ") invalid in logsumexp. Returning max value." << std::endl;
             return max_val;
        }
        return max_val + std::log(sum_exp);
    }
};

} // namespace kws

#endif // KWS_GMM_HMM_H 