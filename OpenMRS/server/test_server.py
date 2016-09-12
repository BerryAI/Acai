import os
import sys
from server import app, om
import tempfile
import unittest
import json


class ServerTest(unittest.TestCase):

    def setUp(self):
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        self.app = app.test_client()

    def tearDown(self):
        os.close(self.db_fd)

    def test_train_cf(self):
        example_ratings = om.data.get_example_ratings()
        res = self.app.post('/train_cf',
                            data=json.dumps(dict(all_ratings=example_ratings)),
                            content_type='application/json')
        self.assertEqual(res.status_code, 200)
        result_data = res.data
        self.assertTrue(len(result_data) > 0)

    def test_train_user_model(self):
        example_ratings = om.data.get_example_ratings()
        res = self.app.post('/train_cf',
                            data=json.dumps(dict(all_ratings=example_ratings)),
                            content_type='application/json')
        hidden_features = json.loads(res.data)
        res = self.app.post('/train_user_model',
                            data=json.dumps(dict(
                                hidden_features=hidden_features,
                                ratings_one_user=example_ratings.values()[0]
                            )),
                            content_type='application/json')
        self.assertEqual(res.status_code, 200)
        self.assertTrue(json.loads(res.data).get('user_model') is not None)

    def test_recommend(self):
        example_ratings = om.data.get_example_ratings()
        res = self.app.post('/train_cf',
                            data=json.dumps(dict(all_ratings=example_ratings)),
                            content_type='application/json')
        hidden_features = json.loads(res.data)
        res = self.app.post('/train_user_model',
                            data=json.dumps(dict(
                                hidden_features=hidden_features,
                                ratings_one_user=example_ratings.values()[0]
                            )),
                            content_type='application/json')
        user_model = json.loads(res.data)['user_model']
        res = self.app.post(
            '/recommend',
            data=json.dumps(dict(
                user_model=user_model,
                hidden_features=hidden_features,
                num_tracks=2
            )),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        tracks = json.loads(res.data)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(isinstance(tracks[0], dict))


if __name__ == '__main__':
    unittest.main()
