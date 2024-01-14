import json


def load_data_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def load_dataset():
    train_path = 'Data/MELD_train_efr.json'
    val_path = 'Data/MELD_val_efr.json'
    test_path = 'Data/MELD_test_efr.json'
    return load_dataset_custom_paths(train_path, val_path, test_path)


def load_dataset_custom_paths(training_path, validation_path, testing_path):
    def extract_fields(data):
        utterances = []
        emotions = []
        emotions_UQ = set()  # using set to get unique
        triggers = []

        for episode in data:
            utterances.extend(episode['utterances'])
            emotions.extend(episode['emotions'])
            emotions_UQ.update(episode['emotions'])
            # triggers.extend(episode['triggers'])

        return {'utterances': utterances, 'emotions': emotions, 'triggers': triggers, 'unique-emotions': emotions_UQ}

    # Load training data
    raw_train_data = load_data_from_json(training_path)
    train_data = extract_fields(raw_train_data)

    # Load validation data
    raw_val_data = load_data_from_json(validation_path)
    val_data = extract_fields(raw_val_data)

    # Load test data
    raw_test_data = load_data_from_json(testing_path)
    test_data = extract_fields(raw_test_data)

    return {'train': train_data, 'val': val_data, 'test': test_data}
