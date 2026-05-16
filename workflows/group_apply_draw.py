import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group_tag")
    parser.add_argument("version_start", type=int)
    parser.add_argument("version_end", type=int)
    parser.add_argument("skip_version", type=int, nargs="?", default=0)
    parser.add_argument("draw_only", type=int, nargs="?", default=0)
    parser.add_argument("dataset_year", nargs="?", default="")
    parser.add_argument("data_input_override", nargs="?", default="")
    parser.add_argument("output_prefix", nargs="?", default="")
    parser.add_argument("fid_profile", nargs="?", default="auto")
    args = parser.parse_args()

    dataset_year = "" if args.dataset_year == "__EMPTY__" else args.dataset_year
    data_input_override = "" if args.data_input_override == "__EMPTY__" else args.data_input_override
    output_prefix = "" if args.output_prefix == "__EMPTY__" else args.output_prefix
    fid_profile = "auto" if args.fid_profile == "__EMPTY__" else args.fid_profile

    tags = []
    for version in range(args.version_start, args.version_end + 1):
        if version == args.skip_version:
            continue
        tags.append(f"{args.group_tag}_v{version}")

    output_tag = f"{args.group_tag}_v{args.version_start}_v{args.version_end}"

    if not args.draw_only:
        apply_cmd = [sys.executable, "workflows/batch_apply_scores.py", "--output-tag", output_tag]
        if dataset_year:
            apply_cmd += ["--dataset-year", dataset_year]
        if data_input_override:
            apply_cmd += ["--data-input", data_input_override]
        if output_prefix:
            apply_cmd += ["--output-prefix", output_prefix]
        apply_cmd += tags
        rc = subprocess.call(apply_cmd)
        if rc != 0:
            raise SystemExit(rc)

    draw_cmd = [sys.executable, "workflows/batch_draw_scores.py", "--output-tag", output_tag]
    if output_prefix:
        draw_cmd += ["--output-prefix", output_prefix]
    if fid_profile:
        draw_cmd += ["--fid-profile", fid_profile]
    draw_cmd += tags
    raise SystemExit(subprocess.call(draw_cmd))


if __name__ == "__main__":
    main()
