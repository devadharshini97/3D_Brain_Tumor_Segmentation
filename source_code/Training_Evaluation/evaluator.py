import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_predictions(model, test_dataloader):
    model.to(device)
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for image, label in test_dataloader:
            image = image.cuda().to(device)
            label = label.cpu().numpy()
            label = np.argmax(label, -1)
            probabilities = model(image)
            test_predictions.extend(np.argmax(probabilities.cpu(), -1))
            test_labels.extend(label)
            del probabilities, label
    return test_predictions, test_labels
