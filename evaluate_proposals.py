from collections import defaultdict
import json

from tqdm import tqdm
import xmltodict

from utils import *


def main():
    # loading proposals
    with open("Proposal Sample/edge_box_proposals_all.json", "r") as f:
        eb_proposals = json.load(f)

    with open("Proposal Sample/selective_search_proposals_all.json", "r") as f:
        ss_proposals = json.load(f)

    proposal_sets = [(eb_proposals, "edge_box_proposals_labeled"),
                     (ss_proposals, "selective_search_proposals_labeled"),
                     ]
    resized_h, resized_w = 252, 252
    iou_threshold = 0.5
    results = defaultdict(list)

    for proposal_set, results_file_name in proposal_sets:
        for img, proposals in tqdm(proposal_set.items()):
            img_name = img.split(".")[0]
            
            with open(f"/dtu/datasets1/02516/potholes/annotations/{img_name}.xml") as f:
                gt_data = xmltodict.parse(f.read())
            
            # retrieving height and width of image
            img_h = int(gt_data["annotation"]["size"]["height"])
            img_w = int(gt_data["annotation"]["size"]["width"])

            # computing the scale
            scale_x = resized_w / img_w
            scale_y = resized_h / img_h

            gt_boxes = gt_data["annotation"]["object"]

            # put gt_boxes in list if there is only a single gt box
            if type(gt_boxes) == dict:
                gt_boxes = [gt_boxes]

            # checking if the IoU between the proposal and any of the gt boxes is above iou_threshold
            for proposal in proposals:
                label = "background"
                for gt_box in gt_boxes:
                    iou = calc_iou(proposal, gt_box["bndbox"], scale_x, scale_y)
                    if iou >= iou_threshold:
                        label = "pothole"
                        break
                
                # adding the proposal and its label to the results dict
                results[img].append({
                    "x_min": proposal["x_min"],
                    "y_min": proposal["y_min"],
                    "x_max": proposal["x_max"],
                    "y_max": proposal["y_max"],
                    "label": label
                })

        # saving the results as a json file
        with open(f"Proposal Sample/{results_file_name}.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()