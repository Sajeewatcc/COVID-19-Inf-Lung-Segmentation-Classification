import os
import random
import shutil

def create_dataset_structure(data_path, positive_paths, negative_paths, train_ratio=0.8, seed=42):

    random.seed(seed)

    dirs = {
        'train_pos': os.path.join(data_path, 'train', 'positive'),
        'train_neg': os.path.join(data_path, 'train', 'negative'),
        'test_pos' : os.path.join(data_path, 'test',  'positive'),
        'test_neg' : os.path.join(data_path, 'test',  'negative'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    def split_and_copy(image_paths, train_dir, test_dir):
        shuffled = image_paths.copy()
        random.shuffle(shuffled)

        split_idx   = int(len(shuffled) * train_ratio)
        train_files = shuffled[:split_idx]
        test_files  = shuffled[split_idx:]

        for f in train_files:
            shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))
        for f in test_files:
            shutil.copy(f, os.path.join(test_dir,  os.path.basename(f)))

        return len(train_files), len(test_files)

    pos_train, pos_test = split_and_copy(positive_paths, dirs['train_pos'], dirs['test_pos'])

    neg_train, neg_test = split_and_copy(negative_paths, dirs['train_neg'], dirs['test_neg'])

    print("Dataset structure created successfully!")
    print(f"\nCovid/")
    print(f"├── train/")
    print(f"│   ├── positive/  ({pos_train} images)")
    print(f"│   └── negative/  ({neg_train} images)")
    print(f"└── test/")
    print(f"    ├── positive/  ({pos_test} images)")
    print(f"    └── negative/  ({neg_test} images)")
