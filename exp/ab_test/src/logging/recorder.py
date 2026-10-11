"""Persist submitted prompts, raw streams, snapshots, and evaluation results."""
import json
from dataclasses import asdict

class Recorder:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def json(self, name, value):
        path = self.directory / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, default=str))
        temp.replace(path)

    def context(self, context):
        self.json("context.json", asdict(context))

    def response(self, number, response, solution):
        prefix = f"round{number}"
        (self.directory / f"{prefix}.jsonl").write_text(response.stdout)
        (self.directory / f"{prefix}.stderr.txt").write_text(response.stderr)
        if solution is not None:
            (self.directory / f"{prefix}.v").write_text(solution)
        self.json(f"{prefix}.response.json", dict(wall_s=response.wall_s,
                  timed_out=response.timed_out, returncode=response.returncode))
