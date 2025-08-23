# (..]

object = "windows" # This is the object the user wants to find in the image range

for i in range(4721,4793):
    # Reasoning operation
    data_reasoning = {
        "DATASET_NUMBER": f"{i:08d}",
        "VOCAB_SECONDLVL": object
    }
    try:
        with open(f"reasoning.py") as file:
            exec(file.read(), data_reasoning, data_reasoning)
    except RuntimeError as e:
        print(e)
        continue

    # Subprocesses for flat and nested operations
    data_subprocesses = {
        "DATASET_NUMBER": f"{i:08d}",
        "SCALE_UP_SIZE": 512,
        "VOCAB_GROUNDTRUTH": data_reasoning['VOCAB_GROUNDTRUTH'],
        "VOCAB_FRSTLVL": data_reasoning['VOCAB_FRSTLVL'],
        "VOCAB_SECONDLVL": object
    }
    try:
        with open("flat.py") as file:
            exec(file.read(), data_subprocesses)
        with open("nested.py") as file:
            exec(file.read(), data_subprocesses)
    except RuntimeError as e:
        print(e)
        continue