import os, json

def read_json(fjson):
        with open(fjson) as f:
            return json.load(f)
        
dict = read_json('free-energy-data.json')



