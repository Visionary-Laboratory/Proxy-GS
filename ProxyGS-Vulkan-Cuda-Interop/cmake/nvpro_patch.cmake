# Auto-patch nvpro_core2 logger.cpp to add missing unistd.h include
set(_logger_path "${nvpro_core2_SOURCE_DIR}/nvutils/logger.cpp")
file(READ "${_logger_path}" _src)
if(_src MATCHES "#include <signal.h>" AND NOT _src MATCHES "#include <unistd.h>")
  string(REPLACE "#include <signal.h>" "#include <signal.h>\n#include <unistd.h>" _out "${_src}")
  file(WRITE "${_logger_path}" "${_out}")
  message(STATUS "Applied nvpro_core2 logger.cpp patch (added unistd.h)")
endif()