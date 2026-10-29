"""
agent.py
The "agent" layer: takes an image, runs it through the classifier,
looks up the disease in the knowledge base, and produces a structured,
farmer-facing recommendation. This is the file your rover/dashboard
will eventually call for each captured frame.

Usage (standalone):
    python src/agent.py --image path/to/leaf.jpg

Usage (as a module, e.g. from a rover control loop):
    from agent import CropAgent
    agent = CropAgent()
    report = agent.diagnose("frame_001.jpg", location={"lat": 12.34, "lon": 56.78})
    print(report["farmer_message"])
"""

import argparse
import json
import os
from datetime import datetime

from inference import DiseaseClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = 0.55  # below this, the agent flags it as "uncertain"


class CropAgent:
    def __init__(self, model_dir=None, kb_path=None):
        model_dir = model_dir or os.path.join(BASE_DIR, "models")
        kb_path = kb_path or os.path.join(BASE_DIR, "disease_remedies.json")
        self.classifier = DiseaseClassifier(model_dir=model_dir)

        with open(kb_path) as f:
            self.kb = json.load(f)

    def _lookup(self, class_name):
        return self.kb.get(class_name, self.kb["_default"])

    def diagnose(self, image_path, location=None, top_k=3):
        """
        Runs the full pipeline: predict -> look up -> build recommendation.

        Returns a dict report that is easy to log, send to an app,
        or convert to speech for the farmer.
        """
        predictions = self.classifier.predict(image_path, top_k=top_k)
        top_class, top_conf = predictions[0]
        info = self._lookup(top_class)

        is_uncertain = top_conf < CONFIDENCE_THRESHOLD
        is_healthy = info.get("disease") == "Healthy"

        if is_uncertain:
            farmer_message = (
                f"I'm not fully confident about this one ({top_conf*100:.0f}% match to "
                f"{info['disease']}). Consider taking a closer photo or having it checked "
                f"manually. Other possibilities: "
                + ", ".join(f"{self._lookup(c)['disease']} ({p*100:.0f}%)" for c, p in predictions[1:])
            )
        elif is_healthy:
            farmer_message = f"This {info['crop']} plant looks healthy. No action needed."
        else:
            farmer_message = (
                f"{info['crop']}: detected {info['disease']} ({top_conf*100:.0f}% confidence). "
                f"Severity: {info['severity']}. {info['remedy']} ({info['urgency']})"
            )

        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image": os.path.basename(image_path),
            "location": location,
            "top_prediction": {"class": top_class, "confidence": top_conf},
            "all_predictions": predictions,
            "crop": info.get("crop"),
            "disease": info.get("disease"),
            "severity": info.get("severity"),
            "remedy": info.get("remedy"),
            "urgency": info.get("urgency"),
            "is_uncertain": is_uncertain,
            "farmer_message": farmer_message,
        }
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test.jpg")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--kb_path", default=None)
    args = parser.parse_args()

    agent = CropAgent(model_dir=args.model_dir, kb_path=args.kb_path)
    report = agent.diagnose(args.image)

    print(json.dumps(report, indent=2))
    print("\n--- Farmer-facing message ---")
    print(report["farmer_message"])


if __name__ == "__main__":
    main()
