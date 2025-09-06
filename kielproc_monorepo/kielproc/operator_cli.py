#!/usr/bin/env python3
"""Dead-simple operator CLI for one-click legacy processing."""

from pathlib import Path
import argparse
import json
import shutil

from kielproc.run_easy import run_easy_legacy, SitePreset
from kielproc.cli import PRESETS


def main() -> None:
    ap = argparse.ArgumentParser(description="One-click legacy → SOP bundle")
    ap.add_argument("src", type=Path)
    ap.add_argument("--site", default="DefaultSite")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    site: SitePreset = PRESETS[a.site]
    run_dir, summary, artifacts = run_easy_legacy(
        a.src, site, output_base=a.out, strict=True
    )

    # bundle outputs for easy hand-off
    bundle = Path(run_dir).with_suffix("")
    zpath = bundle.with_name(bundle.name + "__bundle.zip")
    shutil.make_archive(str(zpath.with_suffix("")), "zip", root_dir=run_dir)

    manifest = {
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "summary": summary,
        "bundle_zip": str(zpath),
    }
    Path(run_dir, "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

