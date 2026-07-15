from pathlib import Path
import fasttext

TRAIN_PATH = "quality_data/quality.train"
VALID_PATH = "quality_data/quality.valid"
MODEL_PATH = Path("models/quality_classifier.bin")


model = fasttext.train_supervised(
    input=TRAIN_PATH,
    epoch=10,
    lr=0.1,
    dim=100,
    wordNgrams=2,
    minCount=2,
    loss="softmax",
    thread=8,
)

print("Validation:", model.test(VALID_PATH))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_PATH))
print(f"Saved model to {MODEL_PATH}")