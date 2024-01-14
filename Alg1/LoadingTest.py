from Loader import load_dataset

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset['train']['unique-emotions'])
    print(dataset['val']['unique-emotions'])
    print(dataset['test']['unique-emotions'])
