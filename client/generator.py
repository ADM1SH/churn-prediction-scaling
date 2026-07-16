"""Event-generator client: simulates several players sending realistic
telemetry event streams to the C++ telemetry server over a TCP socket.

Each simulated player opens one connection and plays through a few
sequential sessions. "Churn" profile players show the classic warning
pattern across their sessions: session length declining, menu-idle time
rising, deaths clustering right before they quit. "Healthy" profile players
stay stable or improve. This gives the telemetry log real learnable signal
instead of pure noise.

Usage: .venv10projects/bin/python client/generator.py [host] [port]
"""
import json
import random
import socket
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

# ponytail: real time compressed to seconds/fractions-of-seconds so the demo
# runs fast; the server only cares about relative timestamps, not wall clock.


def send_event(sock, **fields):
    sock.sendall((json.dumps(fields) + "\n").encode())


def play_session(sock, player_id, session_idx, *, total_duration, deaths, menu_fraction,
                  movement_events, death_cluster_near_end):
    """Sends one session_start ... session_end stream for a player.

    total_duration is carved into an active (movement) span and a menu-idle
    span so the recorded session_duration actually matches total_duration,
    instead of menu time being tacked on afterward.
    """
    menu_time = total_duration * menu_fraction
    active_span = max(total_duration - menu_time, 2.0)

    t = time.time()
    send_event(sock, player_id=player_id, event_type="session_start", timestamp=t)
    time.sleep(0.01)

    for _ in range(movement_events):
        t += active_span / max(movement_events, 1)
        send_event(sock, player_id=player_id, event_type="movement", timestamp=t)

    # Deaths: either clustered near session end (churn signal) or spread evenly.
    for i in range(deaths):
        if death_cluster_near_end:
            death_t = t - (deaths - i) * 0.05  # bunched right before quitting
        else:
            death_t = t - active_span + active_span * (i + 1) / (deaths + 1)
        send_event(sock, player_id=player_id, event_type="death", timestamp=min(max(death_t, t - active_span), t))

    if menu_time > 0:
        send_event(sock, player_id=player_id, event_type="menu_open", timestamp=t)
        t += menu_time
        send_event(sock, player_id=player_id, event_type="menu_close", timestamp=t)

    t += 0.05
    send_event(sock, player_id=player_id, event_type="session_end", timestamp=t)
    time.sleep(0.01)


def simulate_churn_player(sock, player_id, n_sessions=4):
    """Declining total session length, rising menu-idle share, deaths piling up near the end."""
    for i in range(n_sessions):
        progress = i / max(n_sessions - 1, 1)
        total_duration = max(90 - 65 * progress + random.uniform(-4, 4), 15)
        menu_fraction = min(0.05 + 0.6 * progress + random.uniform(-0.03, 0.03), 0.85)
        deaths = int(1 + round(5 * progress)) + random.randint(0, 1)
        movement_events = max(int(30 * (1 - 0.6 * progress)), 5)
        play_session(sock, player_id, i, total_duration=total_duration, deaths=deaths,
                     menu_fraction=max(menu_fraction, 0.02), movement_events=movement_events,
                     death_cluster_near_end=True)


def simulate_healthy_player(sock, player_id, n_sessions=4):
    """Stable/improving session length, low menu-idle share, few well-spread deaths."""
    for i in range(n_sessions):
        total_duration = 90 + random.uniform(-10, 25)
        menu_fraction = random.uniform(0.02, 0.08)
        deaths = random.randint(0, 2)
        movement_events = random.randint(35, 60)
        play_session(sock, player_id, i, total_duration=total_duration, deaths=deaths,
                     menu_fraction=menu_fraction, movement_events=movement_events,
                     death_cluster_near_end=False)


def run_player(player_id, profile):
    with socket.create_connection((HOST, PORT)) as sock:
        if profile == "churn":
            simulate_churn_player(sock, player_id)
        else:
            simulate_healthy_player(sock, player_id)
    print(f"[generator] finished {player_id} ({profile})")


def main():
    random.seed(7)
    players = [
        ("churn_ivy", "churn"),
        ("churn_jax", "churn"),
        ("churn_kai", "churn"),
        ("healthy_ann", "healthy"),
        ("healthy_bo", "healthy"),
        ("healthy_cid", "healthy"),
    ]
    for player_id, profile in players:
        run_player(player_id, profile)
    print(f"[generator] sent events for {len(players)} players to {HOST}:{PORT}")


if __name__ == "__main__":
    main()
