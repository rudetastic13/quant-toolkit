// Low-level civil-calendar date math and schedule generation.
//
// Epoch contract: every day count in this header is days since 1970-01-01
// (the numpy datetime64[D] epoch), matching finance.dates.date.Date.toordinal().
// Hinnant's civil-from-days algorithms use a 0000-03-01 era internally;
// MARCH_EPOCH shifts that origin to 1970-01-01 and never leaks past this file.
//
// Header-only and pybind-free so it can be unit-tested from C++ directly.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace fin::core {

inline constexpr std::int64_t MARCH_EPOCH = 719'468;

struct YMD {
    int y;
    int m;
    int d;
};

// floor division/modulo (C++ integer division truncates toward zero)
inline constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
    return (a >= 0) ? a / b : -((-a + b - 1) / b);
}

inline constexpr std::int64_t floor_mod(std::int64_t a, std::int64_t b) noexcept {
    return a - floor_div(a, b) * b;
}

inline constexpr bool is_leap(int y) noexcept {
    return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

inline constexpr int days_in_month(int y, int m) noexcept {
    constexpr int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return (m == 2 && is_leap(y)) ? 29 : lengths[m - 1];
}

// Hinnant days_from_civil, shifted to the 1970 epoch
inline constexpr std::int64_t days_from_ymd(int y, int m, int d) noexcept {
    const std::int64_t yy = y - (m <= 2 ? 1 : 0);
    const std::int64_t era = floor_div(yy, 400);
    const std::int64_t yoe = yy - era * 400;
    const std::int64_t mp = m + (m <= 2 ? 9 : -3);
    const std::int64_t doy = (153 * mp + 2) / 5 + d - 1;
    const std::int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146'097 + doe - MARCH_EPOCH;
}

// Hinnant civil_from_days, shifted to the 1970 epoch
inline constexpr YMD ymd_from_days(std::int64_t days) noexcept {
    const std::int64_t z = days + MARCH_EPOCH;
    const std::int64_t era = floor_div(z, 146'097);
    const std::int64_t doe = z - era * 146'097;
    const std::int64_t yoe = (doe - doe / 1'460 + doe / 36'524 - doe / 146'096) / 365;
    const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::int64_t mp = (5 * doy + 2) / 153;
    const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    const int m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    const int y = static_cast<int>(yoe + era * 400 + (m <= 2 ? 1 : 0));
    return {y, m, d};
}

// month index (y*12 + m-1) plus a signed month delta, resolved to a day count
// under the effective roll day (-1 EOM, 1..31 clamped to the month length)
inline std::int64_t month_step(std::int64_t month_index, std::int64_t delta, int roll_day) noexcept {
    const std::int64_t mi = month_index + delta;
    const int y = static_cast<int>(floor_div(mi, 12));
    const int m = static_cast<int>(floor_mod(mi, 12)) + 1;
    int d;
    if (1 <= roll_day && roll_day <= 28) {
        d = roll_day;
    } else if (roll_day == -1) {
        d = days_in_month(y, m);
    } else {
        d = std::min(roll_day, days_in_month(y, m));
    }
    return days_from_ymd(y, m, d);
}

// Upper bound on the number of schedule dates, mirrors helpers.guess_array_size
inline std::size_t schedule_capacity(std::int64_t start, std::int64_t end,
                                     int freq_type, int freq_value) noexcept {
    std::int64_t max_dates = 0;
    if (freq_type != -1) {
        max_dates = end - start + 1;
        if (freq_type == 1) {
            max_dates = (max_dates / 28) / freq_value;
        } else if (freq_type == 2) {
            max_dates = max_dates / freq_value;
        }
    }
    return static_cast<std::size_t>(max_dates) + 6;
}

// Generate an ascending schedule of day counts into out, returning the count.
// Mirrors finance.dates.schedules.vectorized.generate_schedule semantics:
//   freq_type: -1 Once (returns {start, end}) | 1 month-based | 2 day-based
//   roll: Roll.value (-1 EOM, 0 infer from the anchor, 1..31 clamped)
//   direction: 0 forward (anchor = first_regular, positive step),
//              else backward (anchor = last_regular, negative step)
// Stub periods come from endpoint injection: start/end are prepended/appended
// when they differ from first_regular/last_regular, and the regular grid is
// snapped to first_regular/last_regular at its boundary.
inline std::size_t generate_schedule(std::int64_t start, std::int64_t end,
                                     int freq_type, int freq_value,
                                     std::int64_t first_regular, std::int64_t last_regular,
                                     int roll, int direction,
                                     std::int64_t* out, std::size_t out_cap) {
    if (!(start <= first_regular && first_regular <= last_regular && last_regular <= end)) {
        throw std::invalid_argument("Dates provided are not in correct order");
    }
    if (freq_type == -1) {
        out[0] = start;
        out[1] = end;
        return 2;
    }
    if (freq_type != 1 && freq_type != 2) {
        throw std::invalid_argument("Frequency type not supported");
    }

    const bool forward = (direction == 0);
    const std::int64_t anchor = forward ? first_regular : last_regular;

    int roll_day = roll;
    std::int64_t anchor_mi = 0;
    if (freq_type == 1) {
        const YMD a = ymd_from_days(anchor);
        anchor_mi = static_cast<std::int64_t>(a.y) * 12 + (a.m - 1);
        if (roll_day == 0) {
            roll_day = (a.d == days_in_month(a.y, a.m)) ? -1 : a.d;
        }
    }

    const auto date_at = [&](std::int64_t k) noexcept {
        return (freq_type == 2) ? anchor + k * freq_value
                                : month_step(anchor_mi, k * freq_value, roll_day);
    };

    std::size_t n = 0;
    const auto push = [&](std::int64_t value) {
        if (n >= out_cap) {
            throw std::length_error("Schedule output buffer too small");
        }
        out[n++] = value;
    };

    if (forward) {
        if (start != first_regular) push(start);
        push(first_regular);
        if (last_regular != first_regular) {
            for (std::int64_t k = 1;; ++k) {
                const std::int64_t d = date_at(k);
                if (d >= last_regular) break;
                push(d);
            }
            push(last_regular);
        }
        if (end != last_regular) push(end);
    } else {
        // build descending, then reverse
        if (end != last_regular) push(end);
        push(last_regular);
        if (first_regular != last_regular) {
            for (std::int64_t k = 1;; ++k) {
                const std::int64_t d = date_at(-k);
                if (d <= first_regular) break;
                push(d);
            }
            push(first_regular);
        }
        if (start != first_regular) push(start);
        std::reverse(out, out + n);
    }
    return n;
}

}  // namespace fin::core
