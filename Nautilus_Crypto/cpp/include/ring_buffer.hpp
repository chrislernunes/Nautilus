#pragma once

#include <atomic>
#include <array>
#include <optional>
#include <cstddef>

namespace nautilus {

// ─────────────────────────────────────────────────────────────────────────────
// Single-Producer Single-Consumer lock-free ring buffer.
//
// Used to pass OHLCVBar objects from the C++ ingestion thread to Python
// without any mutex contention on the hot path.
//
// Template params:
//   T    – element type (must be trivially copyable for ABA-safety)
//   N    – capacity (power-of-2 strongly recommended for masking trick)
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, std::size_t N>
class SPSCRingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

public:
    // Producer: returns false if buffer is full (non-blocking)
    bool push(const T& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire))
            return false;   // buffer full — caller decides what to do
        data_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer: returns nullopt if buffer is empty (non-blocking)
    std::optional<T> pop() noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return std::nullopt;
        T item = data_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return item;
    }

    bool empty() const noexcept {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & mask_;
    }

private:
    static constexpr std::size_t mask_ = N - 1;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::array<T, N>         data_{};
};

} // namespace nautilus
