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