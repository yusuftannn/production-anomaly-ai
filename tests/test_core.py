import json
import io
import unittest
import zipfile
from unittest.mock import patch, MagicMock
import pandas as pd
from core import analyze, validate_data, feature_frame, FEATURES, export_csv, export_report, record_context
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

    def test_per_machine_models_use_training_history_only(self):
        result = analyze(self.df, model_scope='per_machine')
        self.assertEqual(set(result.machine_models), set(result.train.machine))
        self.assertEqual(result.model_scope, 'per_machine')
        self.assertEqual(result.fallback_machines, [])
        changed = self.df.copy()
        changed['is_anomaly'] = 1 - changed.is_anomaly
        other = analyze(changed, model_scope='per_machine')
        pd.testing.assert_series_equal(result.test.anomaly_score, other.test.anomaly_score)
        pd.testing.assert_series_equal(result.test.predicted_anomaly, other.test.predicted_anomaly)

    def test_per_machine_model_falls_back_without_enough_history(self):
        known_machines = self.df[self.df.machine != 'Makine-04']
        late_machine = self.df[self.df.machine == 'Makine-04'].tail(20)
        result = analyze(pd.concat([known_machines, late_machine]), model_scope='per_machine')
        self.assertNotIn('Makine-04', result.machine_models)
        self.assertIn('Makine-04', result.fallback_machines)
        self.assertIn('Makine-04', result.unseen_machines)

    def test_metrics_are_reported_per_machine(self):
        test = self.result.test
        by_machine = self.result.metrics['by_machine']
        self.assertEqual(set(by_machine), set(test['machine']))
        self.assertEqual(sum(item['records'] for item in by_machine.values()), len(test))
        for machine, group in test.groupby('machine'):
            item = by_machine[machine]
            true_positive = ((group['is_anomaly'] == 1) & (group['predicted_anomaly'] == 1)).sum()
            predicted_positive = group['predicted_anomaly'].sum()
            actual_positive = group['is_anomaly'].sum()
            precision = true_positive / predicted_positive if predicted_positive else 0
            recall = true_positive / actual_positive if actual_positive else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            self.assertAlmostEqual(item['precision'], precision)
            self.assertAlmostEqual(item['recall'], recall)
            self.assertAlmostEqual(item['f1'], f1)

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
        for item in ctx['features'].values():
            self.assertIn(item['range_status'], {'below', 'within', 'above'})
            expected = ('below' if item['value'] < item['training_p10'] else
                        'above' if item['value'] > item['training_p90'] else 'within')
            self.assertEqual(item['range_status'], expected)

    def test_report_bundle_contains_results_metrics_and_metadata(self):
        metadata = {'source': 'demo', 'model_scope': 'shared'}
        with zipfile.ZipFile(io.BytesIO(export_report(self.result, metadata))) as archive:
            self.assertEqual(set(archive.namelist()), {
                'test_results.csv', 'metrics.json', 'run_metadata.json'})
            self.assertIn('predicted_anomaly', archive.read('test_results.csv').decode('utf-8-sig'))
            self.assertEqual(json.loads(archive.read('metrics.json')), self.result.metrics)
            self.assertEqual(json.loads(archive.read('run_metadata.json')), metadata)

    def test_csv_formula_guard(self):
        text = export_csv(pd.DataFrame({'machine': ['=1+1', 'normal']})).decode('utf-8-sig')
        self.assertIn("'=1+1", text)

class OllamaTests(unittest.TestCase):
    def response_content(self, content):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'message': {'content': content}}).encode()
        return response

    def response(self, answer):
        return self.response_content(json.dumps(answer))

    def test_schema_and_payload(self):
        answer = {'summary': 'Özet', 'observations': ['Duruş yüksek'], 'checks': ['Kayıtları incele'], 'limitation': 'Teşhis değildir.'}
        with patch('ollama_client.request.urlopen', return_value=self.response(answer)) as call:
            self.assertEqual(explain({'anomaly_score': .1}, 'local-model'), answer)
            payload = json.loads(call.call_args.args[0].data)
            self.assertFalse(payload['stream'])
            self.assertFalse(payload['think'])
            self.assertIn('format', payload)

    def test_json_wrapped_in_markdown_is_accepted(self):
        answer = {'summary': 'Özet', 'observations': [], 'checks': [], 'limitation': 'Teşhis değildir.'}
        content = 'İstenen çıktı:\n```json\n' + json.dumps(answer) + '\n```'
        with patch('ollama_client.request.urlopen', return_value=self.response_content(content)):
            self.assertEqual(explain({}, 'local-model'), answer)

    def test_non_json_content_has_actionable_error(self):
        with patch('ollama_client.request.urlopen', return_value=self.response_content('Üzgünüm, yanıt veremiyorum.')):
            with self.assertRaisesRegex(RuntimeError, 'structured output desteğini kontrol edin'):
                explain({}, 'local-model')

    def test_invalid_schema_rejected(self):
        with patch('ollama_client.request.urlopen', return_value=self.response({'summary': 'Eksik'})):
            with self.assertRaises(RuntimeError):
                explain({}, 'local-model')

if __name__ == '__main__':
    unittest.main()
