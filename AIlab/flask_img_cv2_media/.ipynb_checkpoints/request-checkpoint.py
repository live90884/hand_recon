import requests
import json


req = requests.post('http://127.0.0.1:8891/predict', json = {'image':'https://dg9ugnb21lig7.cloudfront.net/uploads/product_image/13038/1/_250x250_13038_w_250xh_250xfar_C.jpg'})
print(json.loads(req.content))
# {'predictions': 'Lipstick', 'success': True}


req = requests.get('http://127.0.0.1:8891/performance'})
print(json.loads(req.content)["performance"])
#input size: 7500 
#classes number: 31 
#use pretrained: True 
#epochs: 221
#batch size: 128