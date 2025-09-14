START_ID = 4747
END_ID   = 4820
SCALE_UP_SIZE = 512

VOCAB_SECONDLVL = "windows"
VOCAB_GROUNDTRUTH = ["window", "windows", "window frame", "window pane", "windowpane"]

VARIANTS = [
    ("RAW",      None),
    ("NEAREST",  "INTER_NEAREST"),
    ("BILINEAR", "INTER_LINEAR"),
    ("BICUBIC",  "INTER_CUBIC"),
]

for i in range(START_ID, END_ID):
    dataset_number = f"{i:08d}"
    print(f"\n========== DATASET {dataset_number} ==========")

    for variant_name, inter_name in VARIANTS:
        print(f"--- Variant: {variant_name} ---")

        namespace = {
            "DATASET_NUMBER": dataset_number,
            "SCALE_UP_SIZE": SCALE_UP_SIZE,
            "VOCAB_GROUNDTRUTH": VOCAB_GROUNDTRUTH,
            "VOCAB_SECONDLVL": VOCAB_SECONDLVL,
            "UPSAMPLING": variant_name,
            "INTERPOLATION_NAME": inter_name
        }

        try:
            with open("upsampling_experiment/experiment_flat.py", "r", encoding="utf-8") as f:
                code = f.read()
            exec(code, namespace, namespace)

        except FileNotFoundError as e:
            print(f"[ERROR] Missing file: {e}")
            break

        except SystemExit as e:
            print(f"[INFO] Finished (SystemExit): {e}")
            continue

        except Exception as e:
            print(f"[SKIP] Exception: {e}")
            continue
