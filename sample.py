import requests
server = "http://localhost:8080/"
realm = "SMAI"
r = requests.get(f"{server}realms/{realm}/.well-known/openid-configuration")
print(r.status_code)
print(r.json())  # inspect fields like 'issuer' and 'token_endpoint'