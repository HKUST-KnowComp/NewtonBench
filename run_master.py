import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def discover_modules(modules_root: Path) -> List[str]:
    if not modules_root.exists():
        raise FileNotFoundError(f"Modules directory not found: {modules_root}")
    modules = [p.name for p in modules_root.iterdir() if p.is_dir() and p.name.startswith("m")]
    modules.sort()
    if not modules:
        raise RuntimeError("No modules discovered under 'modules/'")
    return modules


def read_models_from_file(models_file: Path) -> List[str]:
    if not models_file.exists():
        return []
    models: List[str] = []
    with models_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    return models


def read_models_from_mapping(repo_root: Path) -> List[str]:
    # Import without executing repository top-level code unnecessarily
    sys.path.insert(0, str(repo_root))
    try:
        from utils.call_llm_api import api_source_mapping  # type: ignore
    except Exception:
        return []
    finally:
        # Do not leave modified sys.path around
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass
    return list(api_source_mapping.keys()) if isinstance(api_source_mapping, dict) else []


def build_commands(
    repo_root: Path,
    modules: Sequence[str],
    models: Sequence[str],
) -> List[List[str]]:
    run_all = repo_root / "run_all_evaluations.py"
    if not run_all.exists():
        raise FileNotFoundError(f"Missing script: {run_all}")
    commands: List[List[str]] = []
    for model in models:
        for module in modules:
            cmd = [
                "python",
                "run_all_evaluations.py",
                "--module",
                module,
                "--model_name",
                model,
                "--no_prompt",
            ]
            commands.append(cmd)
    return commands


def print_commands(commands: Sequence[Sequence[str]], repo_root: Path) -> None:
    # Print portable, human-friendly shell commands for external users.
    # Always show as: python run_all_evaluations.py ... (relative to repo root)
    for cmd in commands:
        # cmd structure: [python_executable, absolute_script_path, ...args]
        shown = ["python", "run_all_evaluations.py", *cmd[2:]]
        # Simple join with quoting only when needed
        parts = []
        for p in shown:
            if any(ch.isspace() for ch in p):
                parts.append(subprocess.list2cmdline([p]))
            else:
                parts.append(p)
        print(" ".join(parts))


def run_commands_parallel(commands: Sequence[Sequence[str]], max_workers: int) -> None:
    def run_cmd(cmd: Sequence[str]) -> int:
        try:
            completed = subprocess.run(cmd, check=False)
            return completed.returncode
        except Exception:
            return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cmd, cmd) for cmd in commands]
        failures = 0
        for fut in concurrent.futures.as_completed(futures):
            rc = fut.result()
            if rc != 0:
                failures += 1
        if failures:
            print(f"Completed with {failures} failures.")


def partition(lst: Sequence[Sequence[str]], k: int) -> List[List[Sequence[str]]]:
    if k <= 0:
        return [list(lst)]
    k = min(k, max(1, len(lst)))
    buckets: List[List[Sequence[str]]] = [[] for _ in range(k)]
    for idx, item in enumerate(lst):
        buckets[idx % k].append(item)
    return buckets


def spawn_mac_terminal_batches(repo_root: Path, batches: List[List[Sequence[str]]]) -> None:
    # macOS Terminal via AppleScript
    # Each batch launches one Terminal window and runs all its commands sequentially
    for batch in batches:
        if not batch:
            continue
        # Build a single shell line: cd repo; cmd1 && cmd2 && ...; echo Done
        parts: List[str] = [f"cd {sh_quote(str(repo_root))}"]
        for cmd in batch:
            parts.append(" ".join(sh_quote(p) for p in cmd))
        parts.append("echo 'Batch completed';")
        full_cmd = "; ".join(parts)
        osa = f'''tell application "Terminal"
    do script "{full_cmd}"
end tell'''
        subprocess.run(["osascript", "-e", osa])


def spawn_windows_terminal_batches(repo_root: Path, batches: List[List[Sequence[str]]]) -> None:
    # Windows: use start cmd.exe /k to open new cmd windows
    # Each window: cd /d repo && cmd1 && cmd2 && ...
    for batch in batches:
        if not batch:
            continue
        parts: List[str] = [f"cd /d {str(repo_root)}"]
        for cmd in batch:
            parts.append(" ".join(cmd))
        chained = " && ".join(parts)
        # start opens a new window; /k keeps it open after commands
        subprocess.run(["cmd", "/c", "start", "cmd", "/k", chained])


def sh_quote(s: str) -> str:
    # Simple shell quoting for AppleScript command content
    if not s:
        return "''"
    if all(c.isalnum() or c in "@%_+=:,./-" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def resolve_models(repo_root: Path, model_name: str, models_file: Path) -> List[str]:
    if model_name:
        return [model_name]
    models = read_models_from_file(models_file)
    if models:
        return models
    models = read_models_from_mapping(repo_root)
    if models:
        return models
    raise RuntimeError(
        "No models resolved. Provide --model_name or a non-empty models file (e.g., configs/models.txt)."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Master runner: build and run evaluation commands across modules/models with parallelism."
        )
    )
    parser.add_argument("-m", "--model_name", type=str, default="", help="Single model to run across all modules.")
    parser.add_argument("-p", "--parallel", type=int, default=5, help="Number of concurrent runs (default 5).")
    parser.add_argument(
        "--models_file",
        type=str,
        default="configs/models.txt",
        help="Path to newline-delimited models list when --model_name is not given.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print the commands that would be run and exit.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    modules = discover_modules(repo_root / "modules")
    models = resolve_models(repo_root, args.model_name, Path(args.models_file))

    commands = build_commands(repo_root, modules, models)

    # Always show the commands before running
    print("Planned commands ({} total):".format(len(commands)))
    print_commands(commands, repo_root)

    if args.print_only:
        return

    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if args.parallel == 1:
        # Sequential, in-process
        run_commands_parallel(commands, max_workers=1)
    else:
        # Default behavior: spawn OS terminals when available; otherwise in-process parallel
        batches = partition(commands, args.parallel)
        if sys.platform == "darwin":
            spawn_mac_terminal_batches(repo_root, batches)
            print(f"Spawned {len(batches)} macOS Terminal window(s). Each runs a batch of commands.")
        elif sys.platform.startswith("win"):
            spawn_windows_terminal_batches(repo_root, batches)
            print(f"Spawned {len(batches)} Windows cmd window(s). Each runs a batch of commands.")
        else:
            run_commands_parallel(commands, max_workers=args.parallel)


if __name__ == "__main__":
    main()
