"""Generates a larger synthetic labeled batch with the same feature schema
as the live telemetry log, so the XGBoost model has enough rows to train on.
The live socket demo (server + generator) proves the ingestion pipeline
works end to end; this batch just gives the classifier real volume.

Schema matches server/telemetry_server.cpp's CSV output exactly:
  player_id,session_duration,movement_events,deaths,menu_opens,menu_time,time_since_last_death
plus a churn_label column (1 = player churned after this session).

ponytail: label is generated from a noisy logistic combination of the
features (not a hard rule), so the classes overlap like real data instead of
being trivially separable.
"""
import numpy as np
import pandas as pd

N_ROWS = 4000
OUT_PATH = "data/training_data.csv"


def main():
    rng = np.random.default_rng(42)

    # Sample raw features from distributions similar to what the generator
    # produces (healthy vs. at-risk mixture), then derive a label from a
    # noisy logistic score so the boundary isn't a clean rule.
    is_at_risk = rng.random(N_ROWS) < 0.45

    session_duration = np.where(
        is_at_risk,
        rng.normal(45, 20, N_ROWS),
        rng.normal(95, 20, N_ROWS),
    ).clip(5, None)

    movement_events = np.where(
        is_at_risk,
        rng.normal(16, 7, N_ROWS),
        rng.normal(45, 12, N_ROWS),
    ).clip(0, None).round().astype(int)

    deaths = np.where(
        is_at_risk,
        rng.poisson(4, N_ROWS),
        rng.poisson(1.2, N_ROWS),
    )

    menu_opens = rng.integers(1, 3, N_ROWS)

    menu_time = np.where(
        is_at_risk,
        rng.normal(22, 10, N_ROWS),
        rng.normal(4, 3, N_ROWS),
    ).clip(0, None)

    time_since_last_death = np.where(
        is_at_risk,
        rng.normal(8, 6, N_ROWS),
        rng.normal(40, 20, N_ROWS),
    ).clip(0, None)

    # Noisy logistic score combining normalized features -> churn probability.
    z = (
        -0.04 * session_duration
        + 0.15 * deaths
        + 0.08 * menu_time
        - 0.03 * time_since_last_death
        - 0.02 * movement_events
        + rng.normal(0, 1.3, N_ROWS)  # noise so it isn't perfectly separable
        + 1.0
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn_label = (rng.random(N_ROWS) < churn_prob).astype(int)

    df = pd.DataFrame({
        "player_id": [f"synthetic_{i}" for i in range(N_ROWS)],
        "session_duration": session_duration.round(2),
        "movement_events": movement_events,
        "deaths": deaths,
        "menu_opens": menu_opens,
        "menu_time": menu_time.round(2),
        "time_since_last_death": time_since_last_death.round(2),
        "churn_label": churn_label,
    })
    df.to_csv(OUT_PATH, index=False)
    print(f"[synthetic_data] wrote {len(df)} rows to {OUT_PATH}, "
          f"churn rate={df['churn_label'].mean():.3f}")


if __name__ == "__main__":
    main()
