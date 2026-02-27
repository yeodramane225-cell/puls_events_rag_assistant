import requests

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"
params = {
    "limit": 10,
    "where": "location_city='Paris'"
}

r = requests.get(url, params=params)
print(r.status_code)
print(r.json())
