import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import uproot

from configs.samples import (
    bd_pbpb_precut_paths,
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_apply_config,
    resolve_fiducial_config,
    resolve_training_config,
    split_root_spec,
    supports_bd_pbpb_precut,
    to_root_spec,
)

DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 3.0
DEFAULT_CACHE_SIZE_MB = 256

ROOT_MACRO_TEMPLATE = r'''
#include <TFile.h>
#include <TObject.h>
#include <TTree.h>
#include <TTreeCache.h>
#include <iostream>
#include <memory>

int {macro_name}(const char* input_path, const char* tree_name, const char* cut_expr, const char* output_path, long long cache_size_bytes) {{
  std::unique_ptr<TFile> input(TFile::Open(input_path, "READ"));
  if (!input || input->IsZombie()) {{
    std::cerr << "ROOT_PRECUT_ERROR open_input " << input_path << std::endl;
    return 10;
  }}

  TTree* tree = dynamic_cast<TTree*>(input->Get(tree_name));
  if (!tree) {{
    std::cerr << "ROOT_PRECUT_ERROR missing_tree " << tree_name << std::endl;
    return 11;
  }}

  if (cache_size_bytes > 0) {{
    tree->SetCacheSize(cache_size_bytes);
    tree->AddBranchToCache("*", true);
    TTreeCache::SetLearnEntries(10);
  }}

  std::unique_ptr<TFile> output(TFile::Open(output_path, "RECREATE"));
  if (!output || output->IsZombie()) {{
    std::cerr << "ROOT_PRECUT_ERROR open_output " << output_path << std::endl;
    return 12;
  }}

  output->cd();
  const char* selection = (cut_expr && cut_expr[0] != '\0') ? cut_expr : "";
  TTree* selected = tree->CopyTree(selection);
  if (!selected) {{
    std::cerr << "ROOT_PRECUT_ERROR copy_tree " << output_path << std::endl;
    return 13;
  }}

  selected->Write(tree_name, TObject::kOverwrite);
  output->Write();
  std::cout << "ROOT_PRECUT_OK entries=" << selected->GetEntries() << std::endl;
  return 0;
}}
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare local precut ROOT inputs for Bd PbPb single-DAG workflows using ROOT-native CopyTree."
    )
    parser.add_argument("train_tag")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument("--cache-size-mb", type=int, default=DEFAULT_CACHE_SIZE_MB)
    parser.add_argument("--force", action="store_true", help="Regenerate precut files even when metadata matches.")
    return parser.parse_args()


def _resolve_source_spec(source_spec: str, input_dir: Path) -> tuple[str, str, str]:
    source_path, tree_name = split_root_spec(source_spec)
    local_candidate = input_dir / Path(source_path).name
    resolved_path = str(local_candidate if local_candidate.exists() else Path(source_path))
    resolved_spec = f"{resolved_path}:{tree_name}"
    return resolved_spec, resolved_path, tree_name


def _to_root_expr(expr: Optional[str]) -> str:
    if expr is None:
        return ""
    out = str(expr).strip()
    if not out:
        return ""
    out = out.replace("&&", " and ").replace("||", " or ")
    out = re.sub(r"\band\b", "&&", out)
    out = re.sub(r"\bor\b", "||", out)
    out = re.sub(r"(?<![=!<>])!(?!=)", "!", out)
    out = re.sub(r"\bnot\b", "!", out)
    chain_pattern = re.compile(
        r"(?P<left>(?:\b[A-Za-z_]\w*\b|-?\d+(?:\.\d+)?))\s*"
        r"(?P<op1><=|>=|<|>)\s*"
        r"(?P<middle>\b[A-Za-z_]\w*\b)\s*"
        r"(?P<op2><=|>=|<|>)\s*"
        r"(?P<right>(?:\b[A-Za-z_]\w*\b|-?\d+(?:\.\d+)?))"
    )

    def _expand_chain(match: re.Match) -> str:
        left = match.group("left")
        op1 = match.group("op1")
        middle = match.group("middle")
        op2 = match.group("op2")
        right = match.group("right")
        reverse_op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
        return f"(({middle} {reverse_op[op1]} {left}) && ({middle} {op2} {right}))"

    previous = None
    while previous != out:
        previous = out
        out = chain_pattern.sub(_expand_chain, out)
    return " ".join(out.split())


def _is_valid_root_output(path: Path, tree_name: str) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                return False
            return root_file[tree_name].num_entries > 0
    except Exception:
        return False


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _source_identity(source_path: str) -> dict:
    identity = {"path": source_path}
    try:
        stat = Path(source_path).stat()
    except OSError:
        return identity
    identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def _expected_metadata(
    resolved_path: str,
    tree_name: str,
    cut_expr: str,
    phase_name: str,
) -> dict:
    provenance = {
        "schema_version": 1,
        "phase": phase_name,
        "source": _source_identity(resolved_path),
        "tree": tree_name,
        "selection": cut_expr,
    }
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    return {**provenance, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def _metadata_matches(output_path: Path, expected: dict) -> bool:
    metadata_path = _metadata_path(output_path)
    try:
        actual = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_metadata(output_path: Path, metadata: dict):
    metadata_path = _metadata_path(output_path)
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(metadata_path)


def _remove_if_exists(path: Path):
    if path.exists():
        path.unlink()


def _write_root_macro() -> Path:
    fd, macro_path = tempfile.mkstemp(prefix="bd_precut_", suffix=".C")
    path = Path(macro_path)
    macro_name = path.stem
    with open(fd, "w") as _:
        pass
    path.write_text(ROOT_MACRO_TEMPLATE.format(macro_name=macro_name))
    return path


def _run_root_copytree(
    macro_path: Path,
    input_path: str,
    tree_name: str,
    cut_expr: str,
    output_path: Path,
    cache_size_mb: int,
):
    call_expr = (
        f'{macro_path.as_posix()}('
        f'{json.dumps(input_path)},'
        f'{json.dumps(tree_name)},'
        f'{json.dumps(cut_expr)},'
        f'{json.dumps(str(output_path))},'
        f'{cache_size_mb * 1024 * 1024}'
        f')'
    )
    return subprocess.run(
        ["root", "-l", "-b", "-q", call_expr],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_phase_with_retry(
    source_spec: str,
    selection_expr: str,
    output_path: Path,
    input_dir: Path,
    max_retries: int,
    retry_delay: float,
    cache_size_mb: int,
    phase_name: str,
    macro_path: Path,
    force: bool,
):
    resolved_spec, resolved_path, tree_name = _resolve_source_spec(source_spec, input_dir)
    _, _ = resolved_spec, tree_name
    cut_expr = _to_root_expr(selection_expr)
    expected_metadata = _expected_metadata(resolved_path, tree_name, cut_expr, phase_name)

    if not force and _is_valid_root_output(output_path, tree_name) and _metadata_matches(output_path, expected_metadata):
        print(f"Using existing precut file: {output_path}", flush=True)
        return

    if output_path.exists():
        reason = "forced regeneration" if force else "invalid output or stale/missing metadata"
        print(f"Removing precut output ({reason}): {output_path}", flush=True)
        _remove_if_exists(output_path)
    _remove_if_exists(_metadata_path(output_path))

    print(f"Creating precut file: {output_path}", flush=True)
    print(f"  Source: {resolved_path}:{tree_name}", flush=True)
    print(f"  Mode: ROOT CopyTree serial read, max_retries={max_retries}, cache_size_mb={cache_size_mb}", flush=True)
    print(f"  Phase: {phase_name}", flush=True)

    last_error = None
    for attempt in range(1, max_retries + 1):
        if output_path.exists():
            _remove_if_exists(output_path)
        result = _run_root_copytree(
            macro_path=macro_path,
            input_path=resolved_path,
            tree_name=tree_name,
            cut_expr=cut_expr,
            output_path=output_path,
            cache_size_mb=cache_size_mb,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            print(stdout, flush=True)
        if stderr:
            print(stderr, flush=True)

        if result.returncode == 0 and _is_valid_root_output(output_path, tree_name):
            with uproot.open(output_path) as root_file:
                entries = root_file[tree_name].num_entries
            _write_metadata(output_path, expected_metadata)
            print(f"  Saved {entries} rows to {output_path}", flush=True)
            return

        last_error = RuntimeError(
            f"ROOT CopyTree failed for {phase_name} on attempt {attempt}/{max_retries} with returncode={result.returncode}"
        )
        print(
            f"  [retry {attempt}/{max_retries}] phase '{phase_name}' failed; output will be retried after cleanup",
            flush=True,
        )
        _remove_if_exists(output_path)
        if attempt < max_retries:
            time.sleep(retry_delay)

    raise last_error or RuntimeError(f"ROOT CopyTree failed for phase '{phase_name}'")


def main():
    args = parse_args()
    train_tag = args.train_tag
    input_dir = Path(args.input_dir)

    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay must be >= 0")
    if args.cache_size_mb < 0:
        raise ValueError("--cache-size-mb must be >= 0")

    if not supports_bd_pbpb_precut(train_tag):
        raise ValueError(
            f"Precut preparation only supports Bd_pb23/Bd_pb24 single-DAG tags, got '{train_tag}'."
        )

    sample = infer_sample_from_tag(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_year = infer_dataset_year(train_tag, sample)
    selection_profile = infer_selection_profile(train_tag, sample)
    fid_profile = infer_fid_profile(train_tag, sample)

    train_cfg = resolve_training_config(sample, channel, dataset_year, selection_profile)
    apply_cfg = resolve_apply_config(sample, channel, dataset_year)
    fid_cfg = resolve_fiducial_config(sample, channel, fid_profile)
    precut_paths = bd_pbpb_precut_paths(train_tag, input_dir=args.input_dir)

    macro_path = _write_root_macro()
    try:
        _run_phase_with_retry(
            source_spec=to_root_spec(train_cfg["background"]),
            selection_expr=train_cfg["background_selection"],
            output_path=precut_paths["train_background"],
            input_dir=input_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            cache_size_mb=args.cache_size_mb,
            phase_name="train_background",
            macro_path=macro_path,
            force=args.force,
        )
        _run_phase_with_retry(
            source_spec=to_root_spec(apply_cfg["data"][0]),
            selection_expr=fid_cfg["expression"],
            output_path=precut_paths["apply_data"],
            input_dir=input_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            cache_size_mb=args.cache_size_mb,
            phase_name="apply_data",
            macro_path=macro_path,
            force=args.force,
        )
    finally:
        _remove_if_exists(macro_path)

    print(f"Prepared precut inputs for {train_tag}", flush=True)
    print(f"  train_background={precut_paths['train_background']}", flush=True)
    print(f"  apply_data={precut_paths['apply_data']}", flush=True)


if __name__ == "__main__":
    main()
