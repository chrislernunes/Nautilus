#pragma once

#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <memory>
#include <chrono>
#include <vector>
#include <cstdint>

namespace nautilus {

// ─────────────────────────────────────────────────────────────────────────────
// Wire-level message types from Binance WebSocket streams
// ─────────────────────────────────────────────────────────────────────────────

struct AggTrade {
    int64_t  event_time;        // E: event time (ms)
    int64_t  trade_id;          // a: agg trade id
    double   price;             // p: price
    double   quantity;          // q: quantity
    int64_t  first_trade_id;    // f: first trade id
    int64_t  last_trade_id;     // l: last trade id
    int64_t  trade_time;        // T: transaction time (ms) ← used for bar building
    bool     is_buyer_maker;    // m: buyer is maker
};

struct KlineData {
    int64_t  open_time;
    double   open;
    double   high;
    double   low;
    double   close;
    double   volume;
    int64_t  close_time;
    double   quote_volume;
    int64_t  num_trades;
    double   taker_buy_base;
    double   taker_buy_quote;
    bool     is_closed;
    std::string interval;
};

// ─────────────────────────────────────────────────────────────────────────────
// 1-second OHLCV bar, built from raw aggTrades
// ─────────────────────────────────────────────────────────────────────────────

struct OHLCVBar {
    int64_t  timestamp_ms;      // bar open time (floor to second)
    double   open;
    double   high;
    double   low;
    double   close;
    double   volume;
    double   buy_volume;        // taker buy volume  (for imbalance feature)
    double   sell_volume;
    int64_t  num_trades;
    double   vwap;
    bool     is_complete;
};

// ─────────────────────────────────────────────────────────────────────────────
// Callbacks
// ─────────────────────────────────────────────────────────────────────────────

using AggTradeCallback  = std::function<void(const AggTrade&)>;
using KlineCallback     = std::function<void(const KlineData&)>;
using OHLCVCallback     = std::function<void(const OHLCVBar&)>;
using StatusCallback    = std::function<void(const std::string&)>;  // log messages

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket client (libwebsockets-based, reconnects with exponential backoff)
// ─────────────────────────────────────────────────────────────────────────────

class WebSocketClient {
public:
    explicit WebSocketClient(
        AggTradeCallback  on_agg_trade,
        KlineCallback     on_kline,
        OHLCVCallback     on_bar,          // fires when 1-second bar is finalised
        StatusCallback    on_status = nullptr
    );
    ~WebSocketClient();

    // Start / stop streaming threads
    void start();
    void stop();

    bool is_connected() const noexcept { return connected_.load(std::memory_order_relaxed); }

    // Expose latest bid/ask from aggTrade stream (approximation)
    double last_price() const noexcept  { return last_price_.load(); }

private:
    void run_loop();
    void connect();
    void on_message(const std::string& payload);
    void parse_agg_trade(const std::string& payload);
    void parse_kline(const std::string& payload);
    void maybe_close_bar(int64_t trade_time_ms, double price, double qty, bool buyer_maker);
    void emit_current_bar(int64_t ts_ms, bool is_complete);

    AggTradeCallback  cb_agg_trade_;
    KlineCallback     cb_kline_;
    OHLCVCallback     cb_bar_;
    StatusCallback    cb_status_;

    std::atomic<bool>   running_   {false};
    std::atomic<bool>   connected_ {false};
    std::atomic<double> last_price_{0.0};

    std::thread         ws_thread_;

    // Current in-progress 1-second bar state
    struct BarAccumulator {
        int64_t bar_ts_ms    = 0;
        double  open         = 0.0;
        double  high         = 0.0;
        double  low          = 1e18;
        double  close        = 0.0;
        double  volume       = 0.0;
        double  buy_volume   = 0.0;
        double  sell_volume  = 0.0;
        double  vwap_num     = 0.0;  // sum(price * qty)
        double  vwap_den     = 0.0;  // sum(qty)
        int64_t num_trades   = 0;
        bool    active       = false;
    } bar_;

    // Reconnection state
    int     backoff_s_ = 1;
    static constexpr int kMaxBackoffS = 64;
};

} // namespace nautilus
