from roboflow import Roboflow

api_key = "Vt1V1kCzLMxzheGkI0WL"

rf = Roboflow(api_key=api_key)
project = rf.workspace("selencakmak").project("tumor-dj2a1")
version = project.version(1)
dataset = version.download("yolov8")
