"""Assemble the final submission zip in the layout the organizers require:

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md

  python package.py <team_name>
"""
import sys
import zipfile
from pathlib import Path

import config

CODE_DIR = Path(__file__).resolve().parents[1]      # code/business_entity_resolution
DOC = CODE_DIR.parents[1] / "Documentation_template.md"


def main(team: str):
    out = CODE_DIR.parents[1] / f"{team}_submission.zip"
    files = {
        config.OUTPUT_DIR / "matching_results.tsv": "output/matching_results.tsv",
        config.OUTPUT_DIR / "candidate_pairs.tsv": "output/candidate_pairs.tsv",
        CODE_DIR / "README.md": "code/business_entity_resolution/README.md",
        CODE_DIR / "requirements.txt": "code/business_entity_resolution/requirements.txt",
        DOC: "Documentation_template.md",
    }
    for p in sorted((CODE_DIR / "src").glob("*.py")):
        files[p] = f"code/business_entity_resolution/src/{p.name}"
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files.items():
            z.write(src, arc)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "team")
