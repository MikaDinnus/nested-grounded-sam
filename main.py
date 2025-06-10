# (..]
for i in range(4656, 4664):
    namespace = {
        "DATASET_NUMBER": str(i),
        "SCALE_UP_SIZE": 512
    }
    try:
        with open("flat.py") as file:
            exec(file.read(), namespace)

        with open("nested.py") as file:
            exec(file.read(), namespace)
    except RuntimeError as e:
        print(e)
        continue