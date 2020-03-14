import csv


def read_csv(file_path):
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)

        _ = next(reader) # Skip first header row
        
        for row in reader:
            assert len(row) == 10 or len(row) == 9
            yield (row[2], row[-1] if len(row) == 10 else None)


if __name__ == "__main__":
    # Test reader
    labeled = list(read_csv("data/sample.csv"))
    unlabeled = list(read_csv("data/thaad_relevant.csv"))

    print(labeled[-1])
    print(unlabeled[-1])