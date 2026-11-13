from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def main():
    # Synthetic teaching dataset. These generated values are not official AQI observations.
    rng = np.random.default_rng(42)
    n = 1500

    max_wind = rng.uniform(2, 35, n)
    max_temp = rng.uniform(20, 45, n)
    min_temp = max_temp - rng.uniform(3, 15, n)
    avg_temp = (max_temp + min_temp) / 2 + rng.normal(0, 1.2, n)
    avg_wind = np.clip(max_wind * rng.uniform(0.25, 0.8, n), 0.5, None)

    # Illustrative relationship for a regression exercise, not a scientific AQI formula.
    aqi = (
        115
        + 2.0 * avg_temp
        - 1.7 * avg_wind
        + 0.9 * max_temp
        - 0.35 * min_temp
        + 0.45 * max_wind
        + rng.normal(0, 12, n)
    )
    aqi = np.clip(aqi, 20, 500)

    df = pd.DataFrame({
        "max_wind_speed": max_wind,
        "max_temperature": max_temp,
        "min_temperature": min_temp,
        "average_temperature": avg_temp,
        "average_wind_speed": avg_wind,
        "aqi": aqi,
    })
    output = DATA_DIR / "aqi_sample.csv"
    df.to_csv(output, index=False)
    print(f"Created {output} with {len(df)} rows.")

if __name__ == "__main__":
    main()
