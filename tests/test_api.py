from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'

def test_prediction():
    payload = {
        'machine_id': 'TEST-001', 'product_type': 'L',
        'air_temperature': 300.0, 'process_temperature': 309.0,
        'rotational_speed': 1300, 'torque': 55.0, 'tool_wear': 210
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['failure_probability'] <= 1
    assert body['risk_level'] in {'LOW','MEDIUM','HIGH','CRITICAL'}
    assert isinstance(body['inspection_items'], list)
