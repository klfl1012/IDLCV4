import json
import random


def calc_iou(proposal: dict, gt_box: dict, scale_x: float, scale_y: float):
    """ 
    Calculates the intersection over union (IoU) between the proposed bounding box and a ground truth bounding box. 

    Parameters: 
    - proposal: Dictionary containing the coordinates of the proposed bounding box.
    - gt_box:   Dictionary containing the coordinates of the ground truth bounding box.
    - scale_x:  Scale to be applied to the x-coordinates.
    - scale_y:  Scale to be applied to the y-coordinates. 

    Returns:
    - iou:      Resulting intersection over union.
    """
    # retrieving coordinates from dictionaries
    x_min_prop, y_min_prop, x_max_prop, y_max_prop = proposal.values()
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = [int(v) for v in gt_box.values()]

    # applying scale to gt coordinates
    x_min_gt *= scale_x
    y_min_gt *= scale_y
    x_max_gt *= scale_x
    y_max_gt *= scale_y

    # calculating overlaps
    x_overlap = max(0, min(x_max_prop, x_max_gt) - max(x_min_prop, x_min_gt))
    y_overlap = max(0, min(y_max_prop, y_max_gt) - max(y_min_prop, y_min_gt))
    intersection = x_overlap * y_overlap

    # calculating sizes of each bounding box
    proposal_size = (x_max_prop - x_min_prop) * (y_max_prop - y_min_prop)
    gt_size = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)

    # calculating union
    union = proposal_size + gt_size - intersection

    # calculating IoU
    iou = intersection / union

    return iou


def filter_proposals(json_path: str, p=0.5, seed=71225) -> None:
    """
    Filters the initially labeled proposals such that a proportion of p proposals are labeled as a pothole.

    Parameters:
    - json_path:    Path to json file containing the initially labeled proposals.
    - p:            Proportion of proposals that should be labeled as pothole. 
    - seed:         Seed for random.sample for reproducibility purposes.
    """
    # setting seed
    random.seed(seed)

    # loading the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_proposals = {}

    for image_name, proposals in data.items():
        # split proposals according to label
        potholes = [p for p in proposals if p["label"] == "pothole"]
        background = [p for p in proposals if p["label"] == "background"]

        n_potholes = len(potholes)

        # if no potholes exist, keep at least one background
        if n_potholes == 0:
            filtered_proposals[image_name] = background[:1]
            continue

        # required number of background samples
        N_b = int(n_potholes * (1 - p) / p)

        # sample background proposals
        sampled_background = random.sample(background, min(N_b, len(background)))

        # combine
        combined = potholes + sampled_background
        random.shuffle(combined)

        filtered_proposals[image_name] = combined

    # save back to same file
    with open(json_path, "w") as f:
        json.dump(filtered_proposals, f, indent=4)

    print(f"Filtered proposals saved to {json_path}")
