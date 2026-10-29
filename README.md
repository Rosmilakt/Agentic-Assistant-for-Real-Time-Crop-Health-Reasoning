# Crop Disease Agent (Software-only scaffold)

Pipeline: **image → disease classifier → knowledge base lookup → farmer recommendation**.
Hardware (rover + camera) is ignored for now — this reads images from disk. When the
rover is ready, you just call `CropAgent.diagnose()` on each captured frame instead of
a file path from argparse.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Get the data

Download the PlantVillage dataset (e.g. from Kaggle: "PlantVillage Dataset").
It usually comes as one folder per class. Split it into train/val:

```bash
python src/split_dataset.py --src /path/to/raw_plantvillage --dst data --val_ratio 0.2
```

This produces:
```
data/train/<ClassName>/*.jpg
data/val/<ClassName>/*.jpg
```

## 3. Train the classifier

```bash
python src/train.py --data_dir data --epochs 15 --batch_size 32
```

- Uses EfficientNet-B0 pretrained on ImageNet, backbone frozen for the first
  `--unfreeze_after` epochs (fast warmup), then fully fine-tuned.
- Saves the best checkpoint to `models/best_model.pt` and the class list to
  `models/classes.json`.
- On a single GPU this is typically 15-25 min for ~15 epochs on the full
  PlantVillage set (~54k images, 38 classes). CPU-only will be much slower —
  consider Google Colab (free GPU) if you don't have a local GPU.

## 4. Run inference on one image

```bash
python src/inference.py --image sample_leaf.jpg
```

## 5. Run the full agent (classifier + remedy recommendation)

```bash
python src/agent.py --image sample_leaf.jpg
```

Output includes a structured JSON report plus a `farmer_message` string —
this is what you'd eventually show on a dashboard, speak aloud, or send via SMS.

## 6. The knowledge base

`src/disease_remedies.json` maps each PlantVillage class name to crop, cause,
symptoms, remedy, severity, and urgency. It currently covers a representative
subset (apple, tomato, potato, corn, grape). **Before your final submission,
extend it to cover all 38 PlantVillage classes** — the folder names in your
`data/train/` directory after splitting are exactly the keys you need to add.
Anything not in the file falls back to a `_default` entry so the agent never crashes.

## Design notes / why it's structured this way

- **Perception and reasoning are decoupled.** `inference.py` only knows about
  pixels → class label. `agent.py` only knows about class label → advice. This
  means you can improve the model (bigger backbone, more data, quantization for
  edge deployment) without touching the recommendation logic, and vice versa.
- **Confidence thresholding.** If the top prediction is below 55% confidence,
  the agent says so instead of confidently giving wrong advice — important for
  a system a farmer will act on.
- **Structured report + natural-language message.** `diagnose()` returns both
  a machine-readable dict (for logging, a map of field health, analytics) and
  a farmer-facing sentence (for the actual interface).

## Where to go next (in rough order)

1. **Fill out the full knowledge base** for all 38 classes.
2. **Evaluate properly**: confusion matrix, per-class precision/recall (PlantVillage
   classes are imbalanced — some diseases have far fewer images).
3. **Field-realism gap**: PlantVillage images are clean lab photos on plain
   backgrounds. Real rover camera images will have background clutter, poor
   lighting, motion blur, multiple leaves overlapping. Two options: (a) collect
   and label a small real-world validation set to measure the drop in accuracy,
   (b) apply heavier augmentation / background-removal preprocessing during
   training to close the gap.
2. **LLM layer on top of the recommendation** (optional but fits your "agent"
   framing well): once you have the structured report from `agent.py`, you can
   pass it to an LLM to turn `remedy` + `urgency` + `severity` into a more
   natural, farmer-friendly, locally-toned explanation, or to answer the
   farmer's follow-up questions ("what if I don't have copper fungicide
   available?"). Keep the classifier and knowledge base as the source of truth
   and use the LLM only for the "explain it well" step, not for diagnosing.
4. **Rover integration hook**: replace the `--image` CLI arg in `agent.py`
   with a call from your camera capture loop, tagging each report with GPS
   coordinates (the `location` param is already there) so you can build a
   field-wide disease heatmap over a session.
5. **Edge deployment**: once trained, export with `torch.jit.trace` or convert
   to ONNX/TensorRT if the rover's onboard compute is limited (e.g. Jetson Nano).
