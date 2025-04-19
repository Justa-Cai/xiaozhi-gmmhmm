#ifndef KWS_MFCC_H
#define KWS_MFCC_H

#include <vector>
#include "kws_sdk.h" // To get MfccParams definition

namespace kws {

/**
 * @brief Extracts MFCC features from a raw audio signal.
 *
 * @param signal The input audio signal (vector of float samples).
 * @param params The MFCC configuration parameters (sample rate, frame length, etc.).
 * @return std::vector<std::vector<double>> A vector of feature vectors.
 *         Each inner vector represents the MFCC features for one frame.
 *         Returns an empty vector if extraction fails or signal is too short.
 */
std::vector<std::vector<double>> extract_mfcc(
    const std::vector<float>& signal,
    const MfccParams& params
);

} // namespace kws

#endif // KWS_MFCC_H 