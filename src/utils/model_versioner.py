import os
import joblib
import datetime
import logging

logger = logging.getLogger(__name__)

class ModelVersioner:
    def __init__(self, base_path='models'):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save_model(self, model, model_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(self.base_path, filename)
        joblib.dump(model, filepath)
        logger.info(f"Saved model: {filepath}")

    def load_latest_model(self, model_name):
        model_files = [f for f in os.listdir(self.base_path) if f.startswith(model_name) and f.endswith('.joblib')]
        if not model_files:
            logger.warning(f"No models found for {model_name}")
            return None
        latest_model = max(model_files)
        filepath = os.path.join(self.base_path, latest_model)
        model = joblib.load(filepath)
        logger.info(f"Loaded model: {filepath}")
        return model

    def get_model_history(self, model_name):
        model_files = [f for f in os.listdir(self.base_path) if f.startswith(model_name) and f.endswith('.joblib')]
        return sorted(model_files, reverse=True)

    def rollback_model(self, model_name, steps=1):
        history = self.get_model_history(model_name)
        if len(history) <= steps:
            logger.warning(f"Not enough versions to rollback {steps} steps")
            return None
        rollback_model = history[steps]
        filepath = os.path.join(self.base_path, rollback_model)
        model = joblib.load(filepath)
        logger.info(f"Rolled back to model: {filepath}")
        return model

# You can add more versioning functionality here as needed