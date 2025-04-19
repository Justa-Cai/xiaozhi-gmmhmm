#include "mfcc.h"
#include <cmath>        // For std::cmath functions like sqrt, log, cos, pow, floor, log10, acos
#include <iostream> // For warnings
#include <vector>
#include <algorithm> // For std::min
#include <numeric>      // For std::inner_product

// Potentially include FFT library headers (e.g., <fftw3.h>)
#include <fftw3.h>

namespace kws {

// Helper function for Discrete Cosine Transform (Type II)
// Takes a vector and returns the first `num_coeffs` DCT coefficients.
std::vector<double> dct_ii(const std::vector<float>& input, size_t num_coeffs) {
    size_t N = input.size();
    if (N == 0 || num_coeffs == 0) return {};
    num_coeffs = std::min(num_coeffs, N);
    std::vector<double> output(num_coeffs);
    double factor = acos(-1.0) / N;
    // Use standard DCT-II scaling (matches common implementations like SciPy)
    double scale = std::sqrt(2.0 / N);
    double scale0 = std::sqrt(1.0 / N);

    for (size_t k = 0; k < num_coeffs; ++k) {
        double sum = 0.0;
        for (size_t n = 0; n < N; ++n) {
            sum += input[n] * std::cos(factor * (n + 0.5) * k);
        }
        // Apply standard DCT-II scaling
        output[k] = sum * ((k == 0) ? scale0 : scale);
    }
    return output;
}

// --- Constants and Helper Functions --- 
const double PI = acos(-1.0);
const float LOG_ENERGY_FLOOR = 1e-12f; // Floor for log energy calculation to avoid log(0)

/**
 * @brief Convert frequency from Hertz to Mel scale.
 */
double hz_to_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

/**
 * @brief Convert frequency from Mel scale to Hertz.
 */
double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

/**
 * @brief Applies a Hamming window to a frame in-place.
 */
void apply_hamming_window(std::vector<float>& frame) {
    size_t N = frame.size();
    if (N == 0) return;
    // Ensure N-1 is not negative or zero if N=1 (though frame len is likely > 1)
    float norm = (N > 1) ? (N - 1) : 1.0f;
    for (size_t n = 0; n < N; ++n) {
        frame[n] *= 0.54f - 0.46f * std::cos(2.0f * PI * n / norm);
    }
}

/**
 * @brief Calculates the Mel filterbank weights.
 */
std::vector<std::vector<float>> get_mel_filterbank(
    int nfilt, int nfft, int sample_rate, double low_freq, double high_freq)
{
    high_freq = (high_freq > 0) ? high_freq : static_cast<double>(sample_rate) / 2.0;
    double low_mel = hz_to_mel(low_freq);
    double high_mel = hz_to_mel(high_freq);

    // Calculate Mel bin centers
    std::vector<double> mel_points(nfilt + 2);
    for (int j = 0; j < nfilt + 2; ++j) {
        mel_points[j] = low_mel + (high_mel - low_mel) * j / (nfilt + 1);
    }

    // Convert Mel points back to Hz and then to FFT bin indices
    std::vector<int> fft_bins(nfilt + 2);
    for (int j = 0; j < nfilt + 2; ++j) {
        fft_bins[j] = static_cast<int>(std::floor((nfft + 1) * mel_to_hz(mel_points[j]) / sample_rate));
    }

    std::vector<std::vector<float>> filterbank(nfilt, std::vector<float>(nfft / 2 + 1, 0.0f));

    for (int j = 0; j < nfilt; ++j) { // For each filter
        int start_bin = fft_bins[j];
        int center_bin = fft_bins[j + 1];
        int end_bin = fft_bins[j + 2];

        // Rising slope
        for (int k = start_bin; k < center_bin; ++k) {
             // Ensure k is within the bounds of the power spectrum array (0 to nfft/2)
            if (k >= 0 && k < filterbank[j].size()) {
                float weight = static_cast<float>(k - start_bin) / (center_bin - start_bin);
                filterbank[j][k] = weight;
            }
        }
        // Falling slope
        for (int k = center_bin; k < end_bin; ++k) {
            if (k >= 0 && k < filterbank[j].size()) { // Bounds check
                 if (end_bin - center_bin != 0) { // Avoid division by zero
                     float weight = static_cast<float>(end_bin - k) / (end_bin - center_bin);
                     filterbank[j][k] = weight;
                 }
            }
        }
    }
    return filterbank;
}

/**
 * @brief Applies Cepstral Liftering.
 */
void apply_liftering(std::vector<double>& cepstra, int L) {
    if (L <= 0 || cepstra.empty()) {
        return;
    }
    double factor = L / 2.0;
    double sin_factor = PI / L;
    for (size_t n = 0; n < cepstra.size(); ++n) {
        // Assuming n is 0-based index, matching python_speech_features implementation detail
        cepstra[n] *= (1.0 + factor * std::sin(sin_factor * n)); 
    }
}

std::vector<std::vector<double>> extract_mfcc(
    const std::vector<float>& signal,
    const MfccParams& params
) {
    std::vector<std::vector<double>> mfcc_features;

    // Validate parameters
    if (params.frame_len <= 0 || params.frame_stride <= 0 || params.sample_rate <= 0 || params.nfft <= 0 || params.num_cep <= 0) {
        std::cerr << "Error: Invalid MFCC parameters provided." << std::endl;
        return mfcc_features;
    }

    int frame_len_samples = static_cast<int>(params.frame_len * params.sample_rate);
    int frame_stride_samples = static_cast<int>(params.frame_stride * params.sample_rate);
    int nfft = params.nfft;

    if (signal.size() < frame_len_samples) {
        // std::cerr << "Warning: Signal too short for even a single frame." << std::endl;
        return mfcc_features;
    }

    // 1. Pre-emphasis
    std::vector<float> emphasized_signal = signal;
    if (params.preemph > 0.0f) {
        for (size_t i = signal.size() - 1; i > 0; --i) {
            emphasized_signal[i] -= params.preemph * emphasized_signal[i - 1];
        }
        emphasized_signal[0] *= (1.0f - params.preemph);
    }

    // Calculate number of frames
    int num_frames = 1 + static_cast<int>((emphasized_signal.size() - frame_len_samples) / frame_stride_samples);
    mfcc_features.reserve(num_frames);

    // --- Prepare FFTW --- 
    // Use fftwf_ for float operations
    std::vector<float> fft_frame(nfft); // Input buffer for FFTW
    fftwf_complex* fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (nfft / 2 + 1));
    fftwf_plan fft_plan = fftwf_plan_dft_r2c_1d(nfft, fft_frame.data(), fft_out, FFTW_ESTIMATE);
    if (!fft_plan) {
        std::cerr << "Error: Failed to create FFTW plan." << std::endl;
        fftwf_free(fft_out);
        return {};
    }

    // --- Precompute Mel Filterbank --- 
    std::vector<std::vector<float>> mel_filters = get_mel_filterbank(
        params.num_mel_filters,
        nfft,
        params.sample_rate,
        params.lowfreq,
        params.highfreq // Assuming params has highfreq, default 0 means sample_rate/2
    );

    // --- Process Frames --- 
    for (int i = 0; i < num_frames; ++i) {
        int start_sample = i * frame_stride_samples;
        std::vector<float> frame(emphasized_signal.begin() + start_sample, 
                                 emphasized_signal.begin() + start_sample + frame_len_samples);

        // Store frame energy *before* windowing if appendEnergy is true
        float frame_log_energy = 0.0f;
        if (params.appendEnergy) {
            float energy = std::inner_product(frame.begin(), frame.end(), frame.begin(), 0.0f);
            frame_log_energy = std::log(std::max(energy, LOG_ENERGY_FLOOR));
        }

        // 3. Windowing
        apply_hamming_window(frame);

        // Prepare frame for FFT (copy and pad with zeros if needed)
        std::fill(fft_frame.begin(), fft_frame.end(), 0.0f);
        std::copy(frame.begin(), frame.end(), fft_frame.begin());

        // 4. FFT
        fftwf_execute(fft_plan);

        // 5. Power Spectrum
        std::vector<float> power_spectrum(nfft / 2 + 1);
        for (int k = 0; k < (nfft / 2 + 1); ++k) {
            power_spectrum[k] = (fft_out[k][0] * fft_out[k][0] + fft_out[k][1] * fft_out[k][1]) / nfft;
        }

        // 6. Mel Filterbank Application
        std::vector<float> filterbank_energies(params.num_mel_filters);
        for (int filt_idx = 0; filt_idx < params.num_mel_filters; ++filt_idx) {
            filterbank_energies[filt_idx] = std::inner_product(
                power_spectrum.begin(), power_spectrum.end(), mel_filters[filt_idx].begin(), 0.0f
            );
        }
        
        // 7. Log Energy
        std::vector<float> log_filterbank_energies(params.num_mel_filters);
        for (int filt_idx = 0; filt_idx < params.num_mel_filters; ++filt_idx) {
             log_filterbank_energies[filt_idx] = std::log(std::max(filterbank_energies[filt_idx], LOG_ENERGY_FLOOR));
        }

        // 8. DCT
        std::vector<double> cepstral_coeffs = dct_ii(log_filterbank_energies, params.num_cep);

        // 9. Append Energy (Replace 0th coefficient)
        if (params.appendEnergy && !cepstral_coeffs.empty()) {
            if (params.num_cep > 0) {
                 cepstral_coeffs[0] = static_cast<double>(frame_log_energy);
            } // else: appendEnergy true but num_cep=0? Edge case, do nothing.
        }

        // 10. Cepstral Liftering
        if (params.ceplifter > 0) {
            apply_liftering(cepstral_coeffs, params.ceplifter);
        }

        mfcc_features.push_back(std::move(cepstral_coeffs));
    }

    // --- Cleanup FFTW --- 
    fftwf_destroy_plan(fft_plan);
    fftwf_free(fft_out);
    // fftwf_cleanup(); // Call this once at the end of your application if needed

    return mfcc_features;
}

} // namespace kws 