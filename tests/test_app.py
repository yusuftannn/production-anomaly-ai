import unittest
import base64
import json
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_demo_training_and_inspection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 7)
        self.assertIn('Şüpheli test kaydı', [m.label for m in app.metric])
        machine = app.multiselect[0].options[0]
        app.multiselect[0].select(machine).run(timeout=30)
        expected = (app.session_state['analysis'].test
                .query('machine == @machine')
                .groupby('timestamp')['anomaly_score'].max().tolist())
        chart = json.loads(app.get('plotly_chart')[0].proto.spec)
        chart_y = chart['data'][0]['y']
        if isinstance(chart_y, dict):
            chart_y = np.frombuffer(base64.b64decode(chart_y['bdata']), dtype=chart_y['dtype']).tolist()
        np.testing.assert_allclose(chart_y, expected)
        app.checkbox[0].uncheck().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.selectbox[0].select(app.selectbox[0].options[-1]).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.slider[0].set_value(.75).run(timeout=30)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.exception), 0)
