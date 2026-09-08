"""Copy supported Nukun workflow nodes to their Oneko IDs; never edit the input."""

import argparse
from copy import deepcopy
import json
from pathlib import Path


ID_MAP = json.loads((Path(__file__).resolve().parents[1] / "migration/node_id_map.json").read_text(encoding="utf-8"))


def migrate_workflow(workflow):
    result = deepcopy(workflow)
    counts = {}
    unsupported = set()

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            key = "class_type" if isinstance(value.get("class_type"), str) else "type" if "id" in value else None
            node_type = value.get(key) if key else None
            if isinstance(node_type, str):
                if node_type in ID_MAP:
                    value[key] = ID_MAP[node_type]
                    counts[node_type] = counts.get(node_type, 0) + 1
                    properties = value.get("properties")
                    if isinstance(properties, dict):
                        if properties.get("Node name for S&R") == node_type:
                            properties["Node name for S&R"] = ID_MAP[node_type]
                        for field in ("cnr_id", "ver", "aux_id"):
                            properties.pop(field, None)
                elif node_type.startswith("Nukun") or node_type == "T5Balancer":
                    unsupported.add(node_type)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(result)
    return result, counts, sorted(unsupported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="New output file; omitted means dry run.")
    args = parser.parse_args()
    workflow = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result, counts, unsupported = migrate_workflow(workflow)
    print(json.dumps({"replacements": counts, "unsupported": unsupported}, indent=2))
    if unsupported:
        print("No file written. Replace unsupported nodes using docs/MIGRATION.md first.")
        return 2
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"Created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
