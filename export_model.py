<<<<<<< HEAD
"""
Export KNN model training data to JSON for browser-side KNN.
"""
import csv
import json

# Load training data
print("Loading training data...")
data = []
with open('alphabet_landmarks.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header
    for row in reader:
        features = [float(x) for x in row[:-1]]
        label = row[-1]
        data.append({
            'features': features,
            'label': label
        })

print(f"Loaded {len(data)} samples")

# Save as JSON
output = {
    'samples': data,
    'k': 5,
    'feature_count': 60
}

with open('docs/model_data.json', 'w') as f:
    json.dump(output, f)

print(f"Exported to docs/model_data.json")
print(f"File size: {len(json.dumps(output)) / 1024:.1f} KB")
=======
"""
Export KNN model training data to JSON for browser-side KNN.
"""
import csv
import json

# Load training data
print("Loading training data...")
data = []
with open('alphabet_landmarks.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header
    for row in reader:
        features = [float(x) for x in row[:-1]]
        label = row[-1]
        data.append({
            'features': features,
            'label': label
        })

print(f"Loaded {len(data)} samples")

# Save as JSON
output = {
    'samples': data,
    'k': 5,
    'feature_count': 60
}

with open('docs/model_data.json', 'w') as f:
    json.dump(output, f)

print(f"Exported to docs/model_data.json")
print(f"File size: {len(json.dumps(output)) / 1024:.1f} KB")
>>>>>>> d59999f20decd792b84a151d542d94ae9043ae56
