import requests

img1_path = '/Users/gtomberlin/Documents/Pictures/google_PNG19633.png'
img2_path = '/Users/gtomberlin/Documents/Pictures/119930_google_512x512.png'

# GET request to hit root endpoint for welcome message
response = requests.get('http://localhost:8000/image_similarity')
print(response.json())

# POST request to make image comparison
url = 'http://localhost:8000/image_similarity/compare'
files = [('images', open(img1_path, 'rb')), ('images', open(img2_path, 'rb'))]
params = {'naed': 1, 'ssim': 0, 'cosine': 1}
response = requests.post(url = url, files = files, params=params)

print(response.json())