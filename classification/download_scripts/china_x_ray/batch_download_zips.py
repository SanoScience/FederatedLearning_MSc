import sharepy
import os

url_base = 'https://sanoscience.sharepoint.com/:x:/r/sites/ResearchProjects/Shared%20Documents/FL_MSc/China%20X_ray/'

files = [
    'Other.zip',
    'Viral.zip',
    'Normal.zip',
    'Dx1_20.zip',
    'Dx21_40.zip',
    'Dx41_51.zip',
    'PE1_10.zip',
    'PE11_20.zip',
    'PE21_23.zip',
]

sp_username = os.environ['SP_USERNAME']
sp_password = os.environ['SP_PASSWORD']

print('SP_USERNAME', sp_username)

s = sharepy.connect("sanoscience.sharepoint.com", username=sp_username, password=sp_password)

for f_name in files:
    r = s.getfile(url_base+f_name, filename=f_name)
    print(f_name, r)
    print(f"{f_name} downloaded")
