import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from core import analyze, validate_data, feature_frame, FEATURES, export_csv, record_context
from demo import generate_demo
from ollama_client import explain

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_demo()
        cls.result = analyze(cls.df)

    def test_temporal_split_and_no_label_leakage(self):
        r = self.result
        self.assertLess(r.train.timestamp.max(), r.test.timestamp.min())
        self.assertEqual(list(r.model.feature_names_in_), FEATURES)
        changed = self.df.copy()
        changed['is_anomaly'] = 1 - changed.is_anomaly
        other = analyze(changed)
        pd.testing.assert_series_equal(r.test.anomaly_score, other.test.anomaly_score)
        pd.testing.assert_series_equal(r.test.predicted_anomaly, other.test.predicted_anomaly)

    def test_synthetic_anomalies_detected(self):
        # Öğretici, belirgin anomalilerde başarısızlık regresyonunu yakala.
        self.assertGreater(self.result.metrics['recall'], .6)
        self.assertGreater(self.result.metrics['precision'], .6)

    def test_bad_counts_and_durations_rejected(self):
        for col, value in [('scrap_count', 9999), ('downtime_minutes', 9999),
                           ('production_count', -1), ('production_count', 1.5), ('shift_minutes', 0)]:
            with self.subTest(col=col, value=value):
                df = self.df.copy()
                df[col] = df[col].astype(float)
                df.loc[0, col] = value
                with self.assertRaises(ValueError):
                    validate_data(df)

    def test_fractional_running_time_is_not_rounded_up(self):
        df = pd.DataFrame({'production_count': [2, 0], 'scrap_count': [0, 0],
                           'downtime_minutes': [9.5, 10], 'shift_minutes': [10, 10]})
        rates = feature_frame(df)['production_per_running_minute'].tolist()
        self.assertEqual(rates, [4.0, 0.0])

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            validate_data(pd.concat([self.df, self.df.iloc[[0]]]))

    def test_labels_optional(self):
        self.assertIsNone(analyze(self.df.drop(columns='is_anomaly')).metrics)

    def test_reference_uses_past_same_machine(self):
        idx = self.result.test.index[0]
        ctx = record_context(self.result, idx)
        self.assertEqual(ctx['baseline_scope'], 'same_machine')
        past = self.result.train_features.loc[self.result.train.machine == ctx['machine']]
        self.assertEqual(ctx['features']['scrap_rate']['training_median'], past.scrap_rate.median())

    def test_csv_formula_guard(self):
        text = export_csv(pd.DataFrame({'machine': ['=1+1', 'normal']})).decode('utf-8-sig')
        self.assertIn("'=1+1", text)

class OllamaTests(unittest.TestCase):
    def response(self, answer):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'message': {'content': json.dumps(answer)}}).encode()
        return response

    def test_schema_and_payload(self):
        answer = {'summary': 'Özet', 'observations': ['Duruş yüksek'], 'checks': ['Kayıtları incele'], 'limitation': 'Teşhis değildir.'}
        with patch('ollama_client.request.urlopen', return_value=self.response(answer)) as call:
            self.assertEqual(explain({'anomaly_score': .1}, 'local-model'), answer)
            payload = json.loads(call.call_args.args[0].data)
            self.assertFalse(payload['stream'])
            self.assertIn('format', payload)

    def test_invalid_schema_rejected(self):
        with patch('ollama_client.request.urlopen', return_value=self.response({'summary': 'Eksik'})):
            with self.assertRaises(RuntimeError):
                explain({}, 'local-model')

if __name__ == '__main__':
    unittest.main()
