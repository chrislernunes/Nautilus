/*
 * bindings.cpp — pybind11 interface exposing the C++ ingestion layer to Python.
 *
 * Design goal: Python sees plain dicts / dataclasses; all hot-path logic
 * (bar-building, ring-buffer management) stays in C++.
 *
 * Build:
 *   cmake -DCMAKE_BUILD_TYPE=Release -S . -B build && cmake --build build
 *   The shared library is installed to python/core/nautilus_cpp.so
 */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include "websocket_client.hpp"
#include "ring_buffer.hpp"

namespace py = pybind11;
using namespace nautilus;

// ─────────────────────────────────────────────────────────────────────────────
// Python-facing thin wrapper around WebSocketClient.
//
// We expose on_message() so the Python websockets bridge can pump raw JSON
// directly into the C++ bar builder when libwebsockets is not compiled.
// ─────────────────────────────────────────────────────────────────────────────

class PyWebSocketClient {
public:
    PyWebSocketClient(py::object on_bar_cb,
                      py::object on_trade_cb,
                      py::object on_kline_cb,
                      py::object on_status_cb)
    {
        // Wrap Python callables into std::function with GIL management
        auto bar_cb = [on_bar_cb](const OHLCVBar& b) {
            py::gil_scoped_acquire gil;
            on_bar_cb(
                b.timestamp_ms,
                b.open, b.high, b.low, b.close,
                b.volume, b.buy_volume, b.sell_volume,
                b.num_trades, b.vwap, b.is_complete
            );
        };

        auto trade_cb = [on_trade_cb](const AggTrade& t) {
            py::gil_scoped_acquire gil;
            on_trade_cb(t.price, t.quantity, t.trade_time, t.is_buyer_maker);
        };

        auto kline_cb = [on_kline_cb](const KlineData& k) {
            py::gil_scoped_acquire gil;
            on_kline_cb(
                k.open_time, k.open, k.high, k.low, k.close,
                k.volume, k.quote_volume, k.num_trades,
                k.taker_buy_base, k.is_closed, k.interval
            );
        };

        StatusCallback status_cb = nullptr;
        if (!on_status_cb.is_none()) {
            status_cb = [on_status_cb](const std::string& msg) {
                py::gil_scoped_acquire gil;
                on_status_cb(msg);
            };
        }

        client_ = std::make_unique<WebSocketClient>(
            std::move(trade_cb),
            std::move(kline_cb),
            std::move(bar_cb),
            std::move(status_cb)
        );
    }

    void start()  { client_->start(); }
    void stop()   { client_->stop(); }
    bool is_connected() const { return client_->is_connected(); }
    double last_price()  const { return client_->last_price(); }

    // Bridge: Python websockets library calls this with raw JSON payload
    void feed_message(const std::string& payload) {
        py::gil_scoped_release release;   // release GIL while C++ processes
        client_->on_message(payload);
    }

private:
    std::unique_ptr<WebSocketClient> client_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(nautilus_cpp, m) {
    m.doc() = "Nautilus BTC — C++ ingestion layer (bar builder + WS client)";

    py::class_<PyWebSocketClient>(m, "WebSocketClient")
        .def(py::init<py::object, py::object, py::object, py::object>(),
             py::arg("on_bar"),
             py::arg("on_trade")  = py::none(),
             py::arg("on_kline")  = py::none(),
             py::arg("on_status") = py::none())
        .def("start",        &PyWebSocketClient::start,
             "Start the WebSocket ingestion thread")
        .def("stop",         &PyWebSocketClient::stop,
             "Stop the WebSocket ingestion thread")
        .def("is_connected", &PyWebSocketClient::is_connected)
        .def("last_price",   &PyWebSocketClient::last_price)
        .def("feed_message", &PyWebSocketClient::feed_message,
             "Feed a raw JSON message from Python WS bridge");

    // Version / build info
    m.attr("__version__") = "1.0.0";
    m.attr("__built_with_lws__") =
#ifdef HAVE_LIBWEBSOCKETS
        true;
#else
        false;
#endif
}
