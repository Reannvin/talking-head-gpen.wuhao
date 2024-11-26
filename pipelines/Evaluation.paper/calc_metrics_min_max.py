import json
import argparse

def find_max_min_values(data, metrics):
    # Initialize dictionaries to store max and min values for each metric
    max_values = {metric: {"video": None, "value": float('-inf')} for metric in metrics}
    min_values = {metric: {"video": None, "value": float('inf')} for metric in metrics}

    # Iterate through each video and update max and min values for each metric
    for video, results in data.items():
        # Skip statistics section
        if video == "statistics":
            continue

        lip_sync = results.get("lip_sync", {})
        fvd = results.get("FVD", None)
        fid = results.get("FID", None)

        # Check and update Offset, LSE-D, LSE-C
        if "Offset" in lip_sync and lip_sync["Offset"] is not None and "Offset" in metrics:
            if lip_sync["Offset"] > max_values["Offset"]["value"]:
                max_values["Offset"]["value"] = lip_sync["Offset"]
                max_values["Offset"]["video"] = video
            if lip_sync["Offset"] < min_values["Offset"]["value"]:
                min_values["Offset"]["value"] = lip_sync["Offset"]
                min_values["Offset"]["video"] = video

        for metric in ["LSE-D", "LSE-C"]:
            if metric in lip_sync and lip_sync[metric] is not None and metric in metrics:
                if lip_sync[metric] > max_values[metric]["value"]:
                    max_values[metric]["value"] = lip_sync[metric]
                    max_values[metric]["video"] = video
                if lip_sync[metric] < min_values[metric]["value"]:
                    min_values[metric]["value"] = lip_sync[metric]
                    min_values[metric]["video"] = video

        # Check and update FID and FVD
        if fid is not None and "FID" in metrics:
            if isinstance(fid, float):  # Ensure fid is a float, not a dict
                if fid > max_values["FID"]["value"]:
                    max_values["FID"]["value"] = fid
                    max_values["FID"]["video"] = video
                if fid < min_values["FID"]["value"]:
                    min_values["FID"]["value"] = fid
                    min_values["FID"]["video"] = video

        if fvd is not None and "FVD" in metrics:
            if isinstance(fvd, float):  # Ensure fvd is a float, not a dict
                if fvd > max_values["FVD"]["value"]:
                    max_values["FVD"]["value"] = fvd
                    max_values["FVD"]["video"] = video
                if fvd < min_values["FVD"]["value"]:
                    min_values["FVD"]["value"] = fvd
                    min_values["FVD"]["video"] = video

    return max_values, min_values


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Find max and min values for specified metrics in video evaluation data.")
    parser.add_argument('--eval-result', type=str, required=True, help='Path to the JSON file with video evaluation results')
    parser.add_argument('--metrics', nargs='+', default=["Offset", "LSE-D", "LSE-C", "FID", "FVD"],
                        help='List of metrics to evaluate (default: Offset, LSE-D, LSE-C, FID, FVD)')

    args = parser.parse_args()

    # Load the evaluation results from the file
    with open(args.eval_result, 'r') as file:
        data = json.load(file)

    # Find the max and min values for the specified metrics
    max_values, min_values = find_max_min_values(data, args.metrics)

    # Display the results
    print("Max values:")
    for metric, info in max_values.items():
        print(f"{metric}: {info['video']} (Value: {info['value']})")

    print("\nMin values:")
    for metric, info in min_values.items():
        print(f"{metric}: {info['video']} (Value: {info['value']})")

if __name__ == "__main__":
    main()
