import os
import sys
import json
from pathlib import Path


def _parse_newroi(img_name):
    name, _ = os.path.splitext(img_name)
    name_split = name.split("_")
    defect_type = _extract_defect_type(name_split).replace(" ", "_")
    timestamp, kb_id, *key_name, light, angle, x1, y1, x2, y2 = name_split
    key_name = "_".join(key_name)
    return {
        "filename": img_name,
        "timestamp": timestamp,
        "kb_id": kb_id,
        "key_name": key_name,
        "light": light,
        "angle": int(angle),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "defectType": defect_type,
        "naming_rule_check": "T",
    }


def _extract_defect_type(namesplit):
    defect_type = []
    while namesplit[-1].isdigit():
        defect_type.append(namesplit.pop())
    while not namesplit[-1].isdigit():
        defect_type.append(namesplit.pop())
    defect_type.reverse()
    return "_".join(defect_type)


if __name__ == "__main__":
	path = Path(sys.argv[1])

	lights = {
		"SolderLight": 0,
		"UniformLight": 1,
		"LowAngleLight": 2,
	}
	labels = []

	for p in path.rglob("*.jpg"):
		attrs = _parse_newroi(p.name)
		label = lights[attrs["light"]]
		labels.append([p.name, label])
	
	with open("dataset.json", "w") as f:
		f.write(json.dumps({"labels": labels}))
