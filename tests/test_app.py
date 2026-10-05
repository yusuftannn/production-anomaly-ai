import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_demo_training_and_inspection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 7)
        self.assertIn('Şüpheli test kaydı', [m.label for m in app.metric])
        app.checkbox[0].uncheck().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.selectbox[0].select(app.selectbox[0].options[-1]).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.slider[0].set_value(.75).run(timeout=30)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.exception), 0)
