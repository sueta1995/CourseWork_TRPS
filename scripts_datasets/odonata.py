# помещается в папку odonata

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))

with open('taxons.json') as file:
    templates = json.load(file)

for item in templates:
    name = item['name']

    os.mkdir(f"{BASE_DIR}\\odonata\\{name}")
