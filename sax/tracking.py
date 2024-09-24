import dataclasses
import json
import os.path
import socket
import sqlite3
import subprocess
import sys
import time
import typing

import beartype

from . import sweep

database_uri = "./experiments.db"


with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as fd:
    schema = fd.read()


@dataclasses.dataclass(frozen=True)
class Experiment:
    id: str
    config: dict[str, sweep.Primitive]
    data: dict[str, object]
    # For reproducibility
    argv: list[str]
    git_hash: str
    git_dirty: bool
    posix_time: int
    hostname: str

    @classmethod
    def new(
        cls, id: str, config: dict[str, sweep.Primitive], data: dict[str, object]
    ) -> typing.Self:
        argv = [sys.executable] + sys.argv
        git_hash = get_git_hash()
        git_dirty = get_git_dirty()
        posix_time = int(time.time())
        hostname = socket.gethostname()
        return cls(id, config, data, argv, git_hash, git_dirty, posix_time, hostname)

    def to_sql_values(self):
        config_json = json.dumps(self.config, separators=(",", ":"))
        data_json = json.dumps(self.data, separators=(",", ":"))
        argv_json = json.dumps(self.argv, separators=(",", ":"))
        return (
            self.id,
            config_json,
            data_json,
            argv_json,
            self.git_hash,
            int(self.git_dirty),
            self.posix_time,
            self.hostname,
        )


def load_by_tag(tag: str) -> list[Experiment]:
    conn = sqlite3.connect(database_uri)
    conn.row_factory = sqlite3.Row
    stmt = "select distinct(experiments.experiment_id), config_json, data_json, argv_json, git_hash, git_dirty, posix_time, hostname from experiments, json_each(json_extract(experiments.config_json, '$.tags')) as tags where tags.value = (?);"
    for row in conn.execute(stmt, (tag,)).fetchall():
        exp_id = row["experiment_id"]
        config = json.loads(row["config_json"])
        data = json.loads(row["data_json"])
        argv = json.loads(row["argv_json"])
        git_dirty = bool(row["git_dirty"])
        yield Experiment(
            exp_id,
            config,
            data,
            argv,
            row["git_hash"],
            git_dirty,
            row["posix_time"],
            row["hostname"],
        )
    conn.close()


@beartype.beartype
def save(
    exp_id: str, config: dict[str, sweep.Primitive], data: dict[str, object]
) -> None:
    exp = Experiment.new(exp_id, config, data)
    stmt = "INSERT INTO experiments(experiment_id, config_json, data_json, argv_json, git_hash, git_dirty, posix_time, hostname) VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
    conn = sqlite3.connect(database_uri)
    conn.execute(schema)
    conn.execute(stmt, exp.to_sql_values())
    conn.commit()
    conn.close()


@beartype.beartype
def get_git_dirty() -> bool:
    try:
        subprocess.check_output([
            "git",
            "diff-index",
            "--quiet",
            "--cached",
            "HEAD",
            "--",
        ])
        subprocess.check_output(["git", "diff-files", "--quiet"])
        return False
    except subprocess.CalledProcessError:
        return True


@beartype.beartype
def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
