"""Veri doğrulama, özellik oluşturma ve zamana göre anomali tespiti."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import io
import zipfile
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

REQUIRED = ['timestamp', 'machine', 'production_count', 'scrap_count', 'downtime_minutes', 'shift_minutes']
FEATURES = ['production_count', 'scrap_rate', 'downtime_rate', 'production_per_running_minute']


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError('Eksik kolonlar: ' + ', '.join(missing))
    if len(frame) < 60:
        raise ValueError('En az 60 kayıt gerekli.')
    if len(frame) > 100_000:
        raise ValueError('İlk sürüm en fazla 100.000 kayıt destekler.')
    df = frame.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    if df['timestamp'].isna().any():
        raise ValueError('Geçersiz veya eksik tarih var; ISO 8601 tarih kullanın.')
    if df['machine'].isna().any() or df['machine'].astype(str).str.strip().eq('').any():
        raise ValueError('Makine adı boş olamaz.')
    df['machine'] = df['machine'].astype(str).str.strip()
    for col in REQUIRED[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'{col}: sonlu ve sıfır veya pozitif sayılar gerekli.')
    for col in ['production_count', 'scrap_count']:
        if (df[col] % 1 != 0).any():
            raise ValueError(f'{col}: adetler tam sayı olmalı.')
    if (df['shift_minutes'] <= 0).any():
        raise ValueError('Vardiya süresi sıfırdan büyük olmalı.')
    if (df['scrap_count'] > df['production_count']).any():
        raise ValueError('Hatalı ürün adedi üretim adedini aşamaz.')
    if (df['downtime_minutes'] > df['shift_minutes']).any():
        raise ValueError('Duruş süresi vardiya süresini aşamaz.')
    if ((df['downtime_minutes'] == df['shift_minutes']) & (df['production_count'] > 0)).any():
        raise ValueError('Vardiyanın tamamı duruşken üretim adedi sıfır olmalı.')
    if df.duplicated(['timestamp', 'machine']).any():
        raise ValueError('Aynı tarih ve makine için yinelenen kayıt var.')
    if 'is_anomaly' in df:
        df['is_anomaly'] = pd.to_numeric(df['is_anomaly'], errors='coerce')
        if not df['is_anomaly'].isin([0, 1]).all():
            raise ValueError('is_anomaly yalnızca 0 veya 1 olmalı; etiketsiz veri için kolonu kaldırın.')
    return df.sort_values(['timestamp', 'machine']).reset_index(drop=True)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x['production_count'] = df['production_count']
    # Sıfır üretimde hata adedi doğrulama gereği sıfırdır.
    x['scrap_rate'] = df['scrap_count'] / df['production_count'].clip(lower=1)
    x['downtime_rate'] = df['downtime_minutes'] / df['shift_minutes']
    running = df['shift_minutes'] - df['downtime_minutes']
    x['production_per_running_minute'] = df['production_count'] / running.mask(running == 0, 1)
    return x[FEATURES]


def machine_metrics(test: pd.DataFrame) -> dict:
    by_machine = {}
    for machine, group in test.groupby('machine', sort=True):
        precision, recall, f1, _ = precision_recall_fscore_support(
            group['is_anomaly'], group['predicted_anomaly'], average='binary', zero_division=0)
        by_machine[str(machine)] = {
            'records': len(group), 'positive_labels': int(group['is_anomaly'].sum()),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        }
    return by_machine


@dataclass
class Analysis:
    train: pd.DataFrame
    test: pd.DataFrame
    train_features: pd.DataFrame
    model: IsolationForest
    machine_models: dict[str, IsolationForest]
    model_scope: str
    fallback_machines: list[str]
    metrics: dict | None
    unseen_machines: list[str]


def analyze(frame: pd.DataFrame, contamination: float = 0.08, train_ratio: float = 0.7,
            model_scope: str = 'shared') -> Analysis:
    if not 0 < contamination <= 0.5:
        raise ValueError('contamination 0 ile 0.5 arasında olmalı (0 hariç).')
    if not 0.5 <= train_ratio <= 0.85:
        raise ValueError('Eğitim oranı 0.5–0.85 arasında olmalı.')
    if model_scope not in {'shared', 'per_machine'}:
        raise ValueError('model_scope shared veya per_machine olmalı.')
    df = validate_data(frame)
    times = df['timestamp'].drop_duplicates().sort_values()
    if len(times) < 10:
        raise ValueError('En az 10 farklı zaman noktası gerekli.')
    cutoff = times.iloc[min(len(times) - 1, max(1, int(len(times) * train_ratio)))]
    train = df[df['timestamp'] < cutoff].copy()
    test = df[df['timestamp'] >= cutoff].copy()
    if len(train) < 30 or len(test) < 10:
        raise ValueError('Zaman ayrımı sonrası en az 30 eğitim, 10 test kaydı gerekli.')
    train_x = feature_frame(train)
    # Tarih, makine adı ve is_anomaly etiketi model özelliklerine dahil edilmez.
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(train_x)
    machine_models = {}
    if model_scope == 'per_machine':
        for machine, group in train.groupby('machine', sort=True):
            if len(group) >= 30:
                machine_models[str(machine)] = IsolationForest(
                    n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
                machine_models[str(machine)].fit(feature_frame(group))
    fallback_machines = sorted(set(test['machine']) - set(machine_models)) if model_scope == 'per_machine' else []
    test['anomaly_score'] = np.nan
    test['predicted_anomaly'] = 0
    for machine, group in test.groupby('machine', sort=True):
        active_model = machine_models.get(str(machine), model)
        test_x = feature_frame(group)
        test.loc[group.index, 'anomaly_score'] = -active_model.decision_function(test_x)
        test.loc[group.index, 'predicted_anomaly'] = (active_model.predict(test_x) == -1).astype(int)
    for col in FEATURES[1:]:
        test[col] = test_x[col]
    metrics = None
    if 'is_anomaly' in test:
        y, p = test['is_anomaly'], test['predicted_anomaly']
        precision, recall, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
        metrics = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                   'confusion_matrix': confusion_matrix(y, p, labels=[0, 1]).tolist(),
                   'positive_labels': int(y.sum()), 'negative_labels': int((y == 0).sum()),
                   'by_machine': machine_metrics(test)}
    unseen = sorted(set(test['machine']) - set(train['machine']))
    return Analysis(train, test, train_x, model, machine_models, model_scope,
                    fallback_machines, metrics, unseen)


def record_context(result: Analysis, index: int) -> dict:
    row = result.test.loc[index]
    same = result.train['machine'].eq(row['machine'])
    baseline = result.train_features.loc[same] if same.any() else result.train_features
    values = feature_frame(result.test.loc[[index]]).iloc[0]
    features = {}
    for key in FEATURES:
        value = float(values[key])
        lower = float(baseline[key].quantile(.1))
        upper = float(baseline[key].quantile(.9))
        features[key] = {
            'value': value,
            'training_median': float(baseline[key].median()),
            'training_p10': lower,
            'training_p90': upper,
            'range_status': 'below' if value < lower else 'above' if value > upper else 'within',
        }
    return {
        'machine': str(row['machine']), 'timestamp': row['timestamp'].isoformat(),
        'predicted_anomaly': bool(row['predicted_anomaly']),
        'anomaly_score': float(row['anomaly_score']),
        'baseline_scope': 'same_machine' if same.any() else 'all_training_machines',
        'baseline_record_count': len(baseline),
        'features': features,
    }


def export_csv(df: pd.DataFrame) -> bytes:
    """Elektronik tablo formülü olarak yorumlanabilecek metinleri etkisizleştir."""
    out = df.copy()
    for col in out.select_dtypes(include=['object', 'string']).columns:
        out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else v)
    return out.to_csv(index=False).encode('utf-8-sig')


def export_report(result: Analysis, metadata: dict | None = None) -> bytes:
    """Package test results, evaluation metrics, and run settings for sharing."""
    report = io.BytesIO()
    with zipfile.ZipFile(report, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('test_results.csv', export_csv(result.test))
        archive.writestr('metrics.json', json.dumps(result.metrics, ensure_ascii=False, indent=2))
        archive.writestr('run_metadata.json', json.dumps(metadata or {}, ensure_ascii=False, indent=2))
    return report.getvalue()
