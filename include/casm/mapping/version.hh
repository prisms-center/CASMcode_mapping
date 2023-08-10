#ifndef CASM_mapping_version
#define CASM_mapping_version

#include <string>

namespace CASM {
namespace mapping {

const std::string &
version();  // Returns the version defined by the CASM_MAPPING_TXT_VERSION
// macro at compile time

}  // namespace mapping
}  // namespace CASM

extern "C" {

/// \brief Return the libcasm_mapping version number
inline const char *casm_mapping_version() {
  return CASM::mapping::version().c_str();
}
}

#endif
