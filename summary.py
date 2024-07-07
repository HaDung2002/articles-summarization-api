import requests

def summary(url, payload, headers):
    response = requests.post(url, json=payload, headers=headers)
    print(response)
    assert response.status_code == 200
    assert "choices" in response.json()
    print(response.json()['choices'][-1]['message']['content'])