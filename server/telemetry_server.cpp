// Plain POSIX-socket TCP telemetry server.
//
// Accepts newline-delimited JSON events from clients (one client == one
// simulated player connection, which may contain several sequential
// sessions). Aggregates events in memory per active session and, on
// session_end (or on connection close mid-session), appends one CSV feature
// row to the telemetry log.
//
// Event schema (flat JSON, one object per line):
//   {"player_id": "p1", "event_type": "movement", "timestamp": 1234.5}
//   event_type in {session_start, session_end, movement, death, menu_open, menu_close}
//
// ponytail: hand-rolled regex JSON parser instead of a JSON library — schema
// is a flat, known set of string/number keys, so a full parser buys nothing.
// Add a real parser (nlohmann/json) if the schema grows nested/array fields.

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>

namespace {

std::mutex g_log_mutex;
std::ofstream g_log_file;

// Parses a flat JSON object line into key -> raw-value-as-string.
// Handles quoted strings and bare numbers; ignores anything else.
std::unordered_map<std::string, std::string> parse_kv(const std::string& line) {
    static const std::regex re(
        R"re("([A-Za-z_]+)"\s*:\s*(?:"([^"]*)"|(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)))re");
    std::unordered_map<std::string, std::string> out;
    for (std::sregex_iterator it(line.begin(), line.end(), re), end; it != end; ++it) {
        auto m = *it;
        std::string key = m[1].str();
        std::string val = m[2].matched ? m[2].str() : m[3].str();
        out[key] = val;
    }
    return out;
}

struct PlayerSession {
    double session_start_ts = 0.0;
    double last_ts = 0.0;
    double last_death_ts = -1.0;
    double pending_menu_open_ts = -1.0;
    double menu_time_accum = 0.0;
    long movement_events = 0;
    long deaths = 0;
    long menu_opens = 0;
};

void write_header_if_empty(const std::string& path) {
    struct stat st{};
    bool need_header = (stat(path.c_str(), &st) != 0) || st.st_size == 0;
    g_log_file.open(path, std::ios::app);
    if (need_header) {
        g_log_file << "player_id,session_duration,movement_events,deaths,menu_opens,"
                       "menu_time,time_since_last_death\n";
        g_log_file.flush();
    }
}

void flush_row(const std::string& player_id, const PlayerSession& s, double end_ts) {
    double duration = end_ts - s.session_start_ts;
    if (duration < 0) duration = 0;
    double time_since_last_death = (s.last_death_ts >= 0) ? (end_ts - s.last_death_ts) : duration;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_file << player_id << ',' << duration << ',' << s.movement_events << ','
               << s.deaths << ',' << s.menu_opens << ',' << s.menu_time_accum << ','
               << time_since_last_death << '\n';
    g_log_file.flush();
    std::cout << "[server] flushed session row for " << player_id
              << " duration=" << duration << " deaths=" << s.deaths
              << " menu_time=" << s.menu_time_accum << "\n";
}

void process_line(const std::string& line, PlayerSession& state, bool& in_session,
                   std::string& current_player_id) {
    auto kv = parse_kv(line);
    auto it_type = kv.find("event_type");
    auto it_pid = kv.find("player_id");
    auto it_ts = kv.find("timestamp");
    if (it_type == kv.end() || it_pid == kv.end() || it_ts == kv.end()) return;

    const std::string& event_type = it_type->second;
    const std::string& player_id = it_pid->second;
    double ts = std::stod(it_ts->second);

    if (event_type == "session_start") {
        state = PlayerSession{};
        state.session_start_ts = ts;
        state.last_ts = ts;
        in_session = true;
        current_player_id = player_id;
    } else if (event_type == "movement") {
        state.movement_events++;
        state.last_ts = ts;
    } else if (event_type == "death") {
        state.deaths++;
        state.last_death_ts = ts;
        state.last_ts = ts;
    } else if (event_type == "menu_open") {
        state.menu_opens++;
        state.pending_menu_open_ts = ts;
        state.last_ts = ts;
    } else if (event_type == "menu_close") {
        if (state.pending_menu_open_ts >= 0) {
            state.menu_time_accum += ts - state.pending_menu_open_ts;
            state.pending_menu_open_ts = -1;
        }
        state.last_ts = ts;
    } else if (event_type == "session_end") {
        if (in_session) {
            flush_row(player_id, state, ts);
            in_session = false;
        }
    }
}

void handle_client(int client_fd) {
    std::string buf;
    PlayerSession state;
    bool in_session = false;
    std::string current_player_id;
    char tmp[4096];

    while (true) {
        ssize_t n = recv(client_fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        buf.append(tmp, static_cast<size_t>(n));
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty()) process_line(line, state, in_session, current_player_id);
        }
    }

    // Client disconnected mid-session (no session_end received): flush what we have
    // so ungraceful disconnects still produce a row.
    if (in_session) {
        flush_row(current_player_id, state, state.last_ts);
    }
    close(client_fd);
}

}  // namespace

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::atoi(argv[1]) : 9000;
    std::string log_path = (argc > 2) ? argv[2] : "data/telemetry_log.csv";

    write_header_if_empty(log_path);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(server_fd, 16) < 0) {
        perror("listen");
        return 1;
    }

    std::cout << "[server] telemetry server listening on port " << port
              << ", logging to " << log_path << "\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }
        std::thread(handle_client, client_fd).detach();
    }
}
