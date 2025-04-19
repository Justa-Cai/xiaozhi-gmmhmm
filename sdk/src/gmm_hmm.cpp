#include "gmm_hmm.h"
#include <stdexcept> // For std::runtime_error in stub if needed
#include <vector>    // Ensure vector is included if needed by removed code
#include <iostream> // Ensure iostream is included if needed by removed code

// This is the implementation file for the GmmHmmModel class.
// It needs to include headers for any libraries used in the scoring logic
// (e.g., Eigen for linear algebra).

// Note: Most implementations were moved into the header (gmm_hmm.h) 
//       as part of the class definition to resolve linking/incomplete type issues.
//       This source file might become empty or only contain non-inline helper functions.

namespace kws {

// --- REMOVE OLD/REDUNDANT DEFINITIONS --- 

// Remove default constructor definition (Error #1)
/* 
GmmHmmModel::GmmHmmModel() { 
    std::cerr << "Warning: GmmHmmModel default constructor called (should not happen)." << std::endl;
}
*/

// Remove score definition (Error #2 - implementation is now in gmm_hmm.h)
/*
double GmmHmmModel::score(const std::vector<std::vector<double>>& features) const {
    // ... implementation was here ... 
}
*/

// Remove get_label definition (Error #3 - implementation is now inline in gmm_hmm.h)
/*
const std::string& GmmHmmModel::get_label() const {
    return label_;
}
*/

// Remove set_label definition (Error #4 - function no longer declared in gmm_hmm.h)
/*
void GmmHmmModel::set_label(const std::string& label) {
    label_ = label;
}
*/

// --- Implement helper methods if any (e.g., calculate_gmm_log_likelihood if it wasn't inline) --- 
// Currently, calculate_gmm_log_likelihood is defined within GmmState in the header.

} // namespace kws 