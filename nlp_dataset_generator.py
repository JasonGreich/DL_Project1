import json
import random

classes = ["person", "car", "cat", "dog", "aeroplane", "bird", "sheep", "horse", "pottedplant", "sofa"]

remove_verbs = [
    "remove", "delete", "eliminate", "erase", "get rid of",
    "take out", "clear", "wipe", "strip", "extract",
    "exclude", "discard", "banish", "expunge", "obliterate",
    "cut", "clean", "get out", "take away", "remove"
]

modifiers = [
    "the", "that", ""
]

articles = [
    "a", "the", ""
]

descriptors = {
    "person": ["tall", "short", "standing", "sitting", "running", "walking", "in the photo", "on the left", "on the right"],
    "car": ["red", "blue", "parked", "moving", "fast", "slow", "in the background", "in the foreground", "vintage"],
    "cat": ["black", "white", "orange", "sleeping", "playing", "fluffy", "cute", "fat"],
    "dog": ["small", "large", "brown", "white", "black", "running", "playing", "cute"],
    "aeroplane": ["flying", "in the sky", "commercial", "small", "large", "jet", "propeller"],
    "bird": ["flying", "perched", "small", "colorful", "singing", "on the branch", "in the tree"],
    "sheep": ["white", "woolly", "grazing", "in the field", "fluffy", "standing"],
    "horse": ["brown", "white", "running", "galloping", "standing", "in the pasture", "tall"],
    "pottedplant": ["green", "small", "on the table", "in the corner", "flowering", "tall"],
    "sofa": ["red", "blue", "leather", "comfortable", "in the living room", "large", "soft"]
}

endings = [
    "from the image",
    "from the picture",
    "from the photo",
    "please",
    "if you can",
    "for me",
    "completely",
    "entirely",
    ""
]

dataset = []

for class_name in classes:
    for i in range(100):
        verb = random.choice(remove_verbs)
        modifier = random.choice(modifiers)
        article = random.choice(articles)
        descriptor = random.choice(descriptors[class_name])
        ending = random.choice(endings)
        
        parts = [verb]
        if modifier:
            parts.append(modifier)
        if article:
            parts.append(article)
        parts.append(descriptor)
        parts.append(class_name)
        if ending:
            parts.append(ending)
        
        sentence = " ".join(parts).strip()
        sentence = sentence.replace("  ", " ")
        
        dataset.append({
            "text": sentence,
            "label": class_name
        })

random.shuffle(dataset)

with open("nlp_training_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} training samples")
print(f"Classes: {classes}")
print(f"Samples per class: {len(dataset) // len(classes)}")
print("\nSample sentences:")
for i in range(5):
    print(f"  {dataset[i]['text']} -> {dataset[i]['label']}")
