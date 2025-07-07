# (..]
for i in range(4608, 4632):
    namespace = {
        "DATASET_NUMBER": str(i),
        "SCALE_UP_SIZE": 512,
        "VOCAB_GROUNDTRUTH": ["window", "windows", "window pane", "pane"],
        "VOCAB_FRSTLVL": "building",
        "VOCAB_SECONDLVL": "windows"
    }
    try:
        with open("flat.py") as file:
            exec(file.read(), namespace)

        with open("nested.py") as file:
            exec(file.read(), namespace)
    except RuntimeError as e:
        print(e)
        continue