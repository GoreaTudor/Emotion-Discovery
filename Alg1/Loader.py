import json


def load_data_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def load_dataset():
    train_path = '../Data/MELD_train_efr.json'
    val_path = '../Data/MELD_val_efr.json'
    test_path = '../Data/MELD_test_efr.json'
    return load_dataset_custom_paths(train_path, val_path, test_path)


def load_dataset_custom_paths(training_path, validation_path, testing_path):
    # Load training data
    train_data = load_data_from_json(training_path)
    train_episodes = train_data['episodes']
    train_speakers = train_data['speakers']
    train_emotions = train_data['emotions']
    train_utterances = train_data['utterances']
    train_triggers = train_data['triggers']

    # Load validation data
    val_data = load_data_from_json(validation_path)
    val_episodes = val_data['episodes']
    val_speakers = val_data['speakers']
    val_emotions = val_data['emotions']
    val_utterances = val_data['utterances']
    val_triggers = val_data['triggers']

    # Load test data
    test_data = load_data_from_json(testing_path)
    test_episodes = test_data['episodes']
    test_speakers = test_data['speakers']
    test_emotions = test_data['emotions']
    test_utterances = test_data['utterances']
    test_triggers = test_data['triggers']

    return (train_episodes, train_speakers, train_emotions, train_utterances, train_triggers), \
           (val_episodes, val_speakers, val_emotions, val_utterances, val_triggers), \
           (test_episodes, test_speakers, test_emotions, test_utterances, test_triggers)
