"""Inference and HTTP integration tests. Created 2026-09-21 KST."""
import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from recognition import Recognizer, preprocess
from train_model import load_data
from web_version.server import Handler

class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = Recognizer()

    def test_blank_rejected(self):
        with self.assertRaises(ValueError):
            preprocess(Image.new('L', (320, 320), 255))

    def test_translation_and_scale(self):
        images = []
        for box in ((10, 10, 30, 110), (180, 50, 220, 250)):
            image = Image.new('L', (320, 320), 255)
            ImageDraw.Draw(image).rectangle(box, fill=0)
            images.append(preprocess(image))
        self.assertEqual(images[0].shape, (28, 28))
        self.assertLess(float(np.abs(images[0] - images[1]).mean()), .02)

    def test_real_mnist_drawings(self):
        dataset = load_data('t10k')
        seen = set()
        correct = 0
        for tensor, label in dataset:
            digit = int(label)
            if digit in seen:
                continue
            seen.add(digit)
            image = ImageOps.invert(Image.fromarray((tensor[0].numpy()*255).astype('uint8'))).resize((280, 280))
            paper = Image.new('L', (320, 320), 255)
            paper.paste(image, (20, 20))
            result = self.recognizer.predict(paper)
            correct += result['digit'] == digit
            self.assertAlmostEqual(sum(result['probabilities']), 1, places=5)
            self.assertEqual(len(result['pixels']), 784)
            if len(seen) == 10:
                break
        self.assertGreaterEqual(correct, 9)

class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.recognizer = Recognizer()
        cls.server = HTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, payload):
        request = urllib.request.Request(self.url + '/predict', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=10)

    def test_home_and_health(self):
        with urllib.request.urlopen(self.url) as response:
            self.assertIn(b'Handwritten Digit Studio', response.read())
        with urllib.request.urlopen(self.url + '/health') as response:
            self.assertTrue(json.load(response)['ready'])

    def test_invalid_payloads(self):
        for payload in ({}, [], {'image':'bad'}, {'image':'data:image/png;base64,!!!'}):
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as exc:
                self.post(payload)
            self.assertEqual(exc.exception.code, 400)

    def test_png_prediction_and_blank(self):
        for blank in (False, True):
            image = Image.new('L', (320, 320), 255)
            if not blank:
                ImageDraw.Draw(image).line((160, 70, 160, 250), fill=0, width=20)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            payload = {'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}
            if blank:
                with self.assertRaises(urllib.error.HTTPError) as exc:
                    self.post(payload)
                self.assertEqual(exc.exception.code, 400)
            else:
                with self.post(payload) as response:
                    result = json.load(response)
                self.assertEqual(result['digit'], 1)

if __name__ == '__main__':
    unittest.main()
