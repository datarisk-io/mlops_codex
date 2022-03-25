(import requests)
(import json)

(defclass Logger [object]
    (defn __init__ [self]
        (setv self.credentials '("user", "password"))
        (setv BASE_URL "http://neomaril.datarisk.net/api")))
    
    (defn health [self]
        (-> f"{self.base_url}/health"
            (requests.get :auth self.credentials
            (.json)))))



(defclass Model [object]
    (defn __init__ [self client id]
        (setv self.client client)
        (setv self.id id)
        (setv.base_url = BASE_URL))
        
    (defn status [self]
        (setv url f"{self.base_url}/status/{self.id}")
        (setv req (requests.get url :auth (self.client.auth))
        (req.json)))
        
    (defn delete [self]
        (setv url f"{self.base_url}/{self.id}")
        (setv req (requests.delete url :auth (self.client.auth)))
        (req.json))
        
    (defn predict [self data]
        (setv url f"{self.base_url}/run/{self.id}")
        (setv model_input { "Input": data })
        (setv req (requests.post url :data (json.dumps model_input) :auth (self.client.auth)))
        (req.json)))

    
