"""Tekrarlanabilir, tamamen sentetik üretim verisi. Gerçek fabrika verisi değildir."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(seed: int = 42, days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        for machine in range(1, 5):
            unusual = rng.random() < 0.08
            downtime = float(np.clip(rng.normal(30, 10), 0, 100))
            rate = rng.normal(1.12 + machine * .025, .07)
            scrap_rate = float(np.clip(rng.normal(.02, .007), 0, .05))
            if unusual:
                kind = rng.integers(0, 3)
                if kind == 0:
                    downtime = float(rng.uniform(140, 260))
                elif kind == 1:
                    scrap_rate = float(rng.uniform(.14, .28))
                else:
                    rate *= float(rng.uniform(.3, .5))
            count = int(max(0, round((480 - downtime) * rate)))
            rows.append({'timestamp': (pd.Timestamp('2026-01-01') + pd.to_timedelta(day, unit='D')).isoformat(),
                         'machine': f'Makine-{machine:02d}', 'production_count': count,
                         'scrap_count': int(round(count * scrap_rate)),
                         'downtime_minutes': round(downtime, 2), 'shift_minutes': 480,
                         'is_anomaly': int(unusual)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    path = Path(__file__).parent / 'data' / 'demo_production.csv'
    path.parent.mkdir(exist_ok=True)
    generate_demo().to_csv(path, index=False)
    print(f'Sentetik veri yazıldı: {path}')
