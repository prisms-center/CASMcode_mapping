#include "casm/mapping/version.hh"

using namespace CASM;
using namespace CASM::mapping;

#ifndef CASM_MAPPING_TXT_VERSION
#define CASM_MAPPING_TXT_VERSION "unknown"
#endif

const std::string &CASM::mapping::version() {
  static const std::string &ver = CASM_MAPPING_TXT_VERSION;
  return ver;
};
