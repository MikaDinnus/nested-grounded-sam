import numpy as np

P2_METRICS = ["PRECISIONBOX", "RECALLBOX", "F1BOX","PRECISIONSEGMENT", "RECALLSEGMENT", "F1SEGMENT","MEANIOUBOX", "MEANIOUSEGMENT", "PAIRS_BOX", "PAIRS_SEGMENT"]

P2_IDEAL = np.array([
    1.0, 1.0, 1.0, # Box
    1.0, 1.0, 1.0, # Segmente
    1.0, 1.0, # IoU
    0.0,# Zeit
    0.0, 0.0]) # Pairs, werden später auf die GT gesetzt

P2_WEIGHTS = np.array([
    1, 1, 1,    # Box
    1, 1, 1,    # Segment
    0.7, 0.7,   # Iou
    0.1, # Zeit
    2, 2]) # Pairs
P2_WEIGHTS = P2_WEIGHTS / P2_WEIGHTS.sum()

P2_REDUNDANCY = np.array([
    0.8, 0.8, 0.8,
    0.8, 0.8, 0.8,
    0.5, 0.5,
    0.1,
    0.5, 0.5])

def calcDP2(namespace, weights=P2_WEIGHTS, redundancy=P2_REDUNDANCY, ideal=P2_IDEAL):
    global P2_METRICS
    metrics = P2_METRICS.copy()
    print(f"metrics: {metrics}")
    print(f"len(metrics): {len(metrics)}")
    print(f"len(ideal): {len(ideal)}")
    print(f"namespace keys: {list(namespace.keys())}")

    ideal[9] = namespace["GT_BOX"]
    ideal[10] = namespace["GT_SEGMENT"]

    if namespace["TYPE"] == "FLAT":
        metrics.append("TIMEFLAT")
    elif namespace["TYPE"] == "NESTED":
        metrics.append("TIMENESTED")
    
    values = np.array([float(namespace[m]) for m in metrics])

    diff_squared = (values - ideal) ** 2
    weighted = weights * redundancy * diff_squared
    distance = np.sqrt(np.sum(weighted))
    print(f"Berechnete Distanz: {distance}")
    print("Einbezogene Metriken:", metrics)
    return distance
