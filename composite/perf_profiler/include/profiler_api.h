#pragma once

#include <string>
#include <functional>

namespace jcore::profiler
{

  bool Init() noexcept;
  void Shutdown() noexcept;

  /**
   * Profile a named region.
   * Safe for nested and parallel invocation.
   */
  void ProfileRegion(const std::string &name,
                     const std::function<void()> &fn) noexcept;

  /**
   * Export collected telemetry as JSON (caller owns returned string).
   */
  char *ExportJSON() noexcept;

} // namespace jcore::profiler
