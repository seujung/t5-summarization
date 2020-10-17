import os
import gdown

os.system("rm -rf binary")
os.system("mkdir binary")
url = "https://drive.google.com/uc?id=1xTs1XZi7k0wTsMmfdA5qIcYkdmy-l24b"
output = './binary/config.json'
gdown.download(url, output, quiet=False)


url = "https://drive.google.com/uc?id=11MpFXsn-Kc4wFGQwRwpElsT_bOgCAD0Y"
output = './binary/pytorch_model.bin'
gdown.download(url, output, quiet=False)