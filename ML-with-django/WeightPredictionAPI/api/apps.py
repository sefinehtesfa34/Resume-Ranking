import os
from django.apps import AppConfig
from django.conf import settings
import joblib
class ApiConfig(AppConfig):
    name = 'api'
    MODEL_FILE=os.path.join(settings.MODELS,'WeightPredictionLinRegModel.joblib')
    model=joblib.load(MODEL_FILE)
    

