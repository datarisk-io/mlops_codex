import json 

def process(data, base_path):
    data = json.loads(data)
    
    return { 'result' : str(data) }