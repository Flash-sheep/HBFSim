#pragma once
// Build-only compatibility for installed Boost 1.74: Process is v1 by default.
#include <boost/version.hpp>
#if BOOST_VERSION != 107400
#error This isolated compatibility header is restricted to audited Boost 1.74
#endif
#include <boost/process.hpp>
