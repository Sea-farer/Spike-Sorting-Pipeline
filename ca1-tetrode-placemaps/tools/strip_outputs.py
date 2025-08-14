import json
import sys
from pathlib import Path


def strip_outputs(ipynb_path: Path) -> bool:
    try:
        data = json.loads(ipynb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read {ipynb_path}: {e}")
        return False

    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
    # Common metadata that can contain execution traces
    md = data.get("metadata", {})
    for k in ["widgets", "language_info", "kernelspec"]:
        if k in md:
            # keep kernelspec/language_info to preserve kernel selection but remove display names that may contain paths
            if isinstance(md[k], dict):
                for subk in list(md[k].keys()):
                    if "path" in subk.lower() or "argv" in subk.lower():
                        md[k].pop(subk, None)
            else:
                pass
    data["metadata"] = md

    if changed:
        ipynb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main(argv):
    if len(argv) > 1:
        files = [Path(a) for a in argv[1:]]
    else:
        files = list(Path("notebooks").glob("*.ipynb"))
    total = 0
    for f in files:
        if not f.exists():
            print(f"Skip missing: {f}")
            continue
        if strip_outputs(f):
            total += 1
            print(f"Stripped outputs: {f}")
        else:
            print(f"No outputs to strip: {f}")
    print(f"Done. Modified {total} file(s).")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
