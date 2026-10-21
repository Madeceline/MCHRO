"""Lossless readers for published SND / multicommodity-flow benchmarks.

Unsupported capacity steps and commodity-specific arc restrictions are rejected,
not silently dropped. Archive members are read in memory, never extracted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tarfile
import zipfile

from .evaluation import validate_instance

CANAD_URLS = {family: f"https://commalab.di.unipi.it/files/Data/MMCF/{family}.tgz"
              for family in ("R", "C", "CPlus")}
SOURCE_URL = CANAD_URLS["R"]  # Backward-compatible public constant.
HOLMBERG_URL = "https://users.mai.liu.se/kajho48/problemdata/cnd/cexall.zip"
FNSS_URL = "https://commalab.di.unipi.it/files/Data/MMCF/waxman.tgz"
MNETGEN_FORMAT_SOURCE = "https://commalab.di.unipi.it/files/Data/MMCF/Mnetgen.tgz"
MAX_MEMBER_BYTES = 50_000_000


def _tar_bytes(bundle: tarfile.TarFile, name: str) -> bytes:
    member = bundle.getmember(name)
    if not member.isfile() or member.size > MAX_MEMBER_BYTES:
        raise ValueError("Benchmark member must be a regular file under 50 MB")
    with bundle.extractfile(member) as handle:
        return handle.read()


def _zip_bytes(bundle: zipfile.ZipFile, name: str) -> bytes:
    member = bundle.getinfo(name)
    if member.is_dir() or member.file_size > MAX_MEMBER_BYTES:
        raise ValueError("Benchmark member must be a file under 50 MB")
    return bundle.read(member)


def _rows(raw: bytes) -> list[list[str]]:
    return [line.split() for line in raw.decode("ascii").splitlines() if line.strip()]


def _instance(nodes: list[int], arcs: list[dict], commodities: list[dict],
              metadata: dict) -> dict:
    data = dict(format="static-snd-v2", nodes=nodes, arcs=arcs,
                commodities=commodities, metadata=metadata)
    validate_instance(data)
    return data


def load_canad_benchmark(archive: str | Path, name: str, *, family: str | None = None) -> dict:
    """Read a regular member without extracting files or changing problem data.

    Format: n m k; m rows of tail/head/unit_cost/capacity/fixed_cost/unused/unused;
    k rows of origin/destination/demand. Optional MULTIGEN.DAT: header is ignored.
    Arc IDs are assigned in file order; original node numbers are preserved.
    """
    family = family or Path(archive).name.removesuffix(".tgz")
    if family not in CANAD_URLS:
        raise ValueError("Specify Canad family R, C or CPlus for a renamed archive")
    with tarfile.open(archive, "r:gz") as bundle:
        raw = _tar_bytes(bundle, name)
    lines = [line.strip() for line in raw.decode("ascii").splitlines() if line.strip()]
    if lines and lines[0] == "MULTIGEN.DAT:":
        lines.pop(0)
    if not lines:
        raise ValueError("Empty Canad instance")
    n, m, k = map(int, lines[0].split())
    if min(n, m, k) <= 0 or len(lines) != 1 + m + k:
        raise ValueError("Benchmark counts do not match data rows")
    arcs = []
    for index, line in enumerate(lines[1:1 + m]):
        fields = line.split()
        if len(fields) != 7:
            raise ValueError("Expected seven fields per Canad arc")
        tail, head = map(int, fields[:2])
        cost, capacity, fixed = map(float, fields[2:5])
        arcs.append(dict(id=index, tail=tail, head=head, variable_cost=cost,
                         capacity=capacity, fixed_cost=fixed))
    commodities = []
    for index, line in enumerate(lines[1 + m:]):
        origin, destination, demand = line.split()
        commodities.append(dict(id=index, origin=int(origin), destination=int(destination),
                                demand=float(demand)))
    return _instance(list(range(1, n + 1)), arcs, commodities, {
        "benchmark": f"Canad {family} / CFG01", "instance_name": name,
        "source_url": CANAD_URLS[family], "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model": "directed_splittable_fixed_charge_multicommodity",
        "comparison_group": "fixed_charge_snd", "adaptation": "lossless_format_only",
        "citation": "Crainic, Frangioni & Gendron (2001), Discrete Applied Mathematics 112, 73–99",
    })


def load_holmberg_benchmark(archive: str | Path, name: str) -> dict:
    """Read Holmberg–Yuan's directed, single-step capacitated network design data.

    Free-capacity-only links become zero-fixed-cost arcs; paid links must have
    exactly one capacity step and no free capacity. More general staircase costs
    cannot be merged into a single binary arc and are explicitly rejected.
    Original node counts are obtained from the publisher's 00index1 file.
    """
    with zipfile.ZipFile(archive) as bundle:
        raw = _zip_bytes(bundle, name)
        index_raw = _zip_bytes(bundle, "00index1")
    index = {}
    for line in index_raw.decode("ascii").splitlines():
        if line.strip():
            key, values = line.strip().split(":", 1)
            index[key.rsplit("/", 1)[-1]] = tuple(map(int, values.split("-")[:3]))
    n, expected_arcs, expected_commodities = index[name]
    rows = _rows(raw)
    arcs, commodities = [], []
    position = 0
    while position < len(rows):
        fields = rows[position]
        position += 1
        if len(fields) != 5:
            raise ValueError("Expected five fields per Holmberg link")
        if all(float(x) == 0 for x in fields):
            break
        tail, head, steps = int(fields[0]), int(fields[1]), int(fields[4])
        cost, free_capacity = map(float, fields[2:4])
        if free_capacity < 0 or steps not in (0, 1):
            raise ValueError("Unsupported Holmberg capacity steps")
        if steps:
            if free_capacity != 0:
                raise ValueError("Cannot merge free capacity and a paid capacity step")
            if position >= len(rows) or len(rows[position]) != 2:
                raise ValueError("Missing Holmberg fixed cost / capacity row")
            fixed, capacity = map(float, rows[position])
            position += 1
        else:
            fixed, capacity = 0.0, free_capacity
        arcs.append(dict(id=len(arcs), tail=tail, head=head, variable_cost=cost,
                         capacity=capacity, fixed_cost=fixed))
    else:
        raise ValueError("Missing Holmberg link terminator")
    while position < len(rows):
        fields = rows[position]
        position += 1
        if len(fields) != 4:
            raise ValueError("Expected four fields per Holmberg demand")
        if all(float(x) == 0 for x in fields):
            break
        identifier, origin, destination = map(int, fields[:3])
        commodities.append(dict(id=identifier, origin=origin, destination=destination,
                                demand=float(fields[3])))
    else:
        raise ValueError("Missing Holmberg demand terminator")
    if position != len(rows) or len(arcs) != expected_arcs or len(commodities) != expected_commodities:
        raise ValueError("Holmberg counts / trailing data do not match 00index1")
    return _instance(list(range(1, n + 1)), arcs, commodities, {
        "benchmark": "Holmberg–Yuan CND", "instance_name": name,
        "source_url": HOLMBERG_URL, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "model": "directed_splittable_fixed_charge_multicommodity",
        "comparison_group": "fixed_charge_snd", "adaptation": "lossless_single_step_capacity",
        "citation": "Holmberg & Yuan (2000), Operations Research 48, 461–481; "
                    "Holmberg & Yuan (1998), International Transactions in Operational Research 5, 529–539",
    })


def load_mnetgen_benchmark(archive: str | Path, name: str, *, prefix: str | None = None) -> dict:
    """Read a published homogeneous-arc Mnetgen instance.

    These have NO fixed charges: f=0 is the original linear-flow special case,
    not an invented fixed-charge benchmark. Each commodity must have one source
    and one sink. Commodity-specific arcs, shared multi-arc resources, and binding
    individual-commodity bounds are unsupported and cause an explicit error.
    """
    archive = Path(archive)
    # CanadN/gridmnet/planarmnet archives use the same four-file Mnetgen layout;
    # the archive stem is also their top-level directory in the published bundles.
    prefix = prefix or archive.name.removesuffix(".tar.gz").removesuffix(".tgz")
    if prefix == "waxman":
        prefix = "waxman"
    with tarfile.open(archive, "r:gz") as bundle:
        files = {ext: _tar_bytes(bundle, f"{prefix}/{name}.{ext}")
                 for ext in ("nod", "arc", "mut", "sup")}
    header = _rows(files["nod"])
    # Published bundles use either one tabular row or four one-value lines.
    if len(header) == 1 and len(header[0]) == 4:
        header_values = header[0]
    elif len(header) == 4 and all(len(row) == 1 for row in header):
        header_values = [row[0] for row in header]
    else:
        raise ValueError("Expected four Mnetgen problem-description values")
    k, n, m, bounded = map(int, header_values)
    if min(k, n, m) <= 0:
        raise ValueError("Mnetgen counts must be positive")
    mutual = {}
    for fields in _rows(files["mut"]):
        if len(fields) != 2 or int(fields[0]) in mutual:
            raise ValueError("Invalid / duplicate Mnetgen mutual capacity")
        mutual[int(fields[0])] = float(fields[1])
    if len(mutual) != bounded:
        raise ValueError("Mnetgen mutual-capacity count mismatch")
    arcs, pointers = [], set()
    for fields in _rows(files["arc"]):
        if len(fields) != 7:
            raise ValueError("Expected seven fields per Mnetgen arc")
        identifier, tail, head, commodity = map(int, fields[:4])
        cost, individual = map(float, fields[4:6])
        pointer = int(fields[6])
        if commodity != -1 or pointer not in mutual or pointer in pointers:
            raise ValueError("Unsupported commodity-specific arc or shared multi-arc resource")
        capacity = mutual[pointer]
        if not math.isfinite(individual) or (individual >= 0 and individual < capacity):
            raise ValueError("Binding individual-commodity capacities are not supported")
        pointers.add(pointer)
        arcs.append(dict(id=identifier, tail=tail, head=head, variable_cost=cost,
                         capacity=capacity, fixed_cost=0.0))
    if len(arcs) != m or pointers != set(mutual):
        raise ValueError("Mnetgen arc count / capacity pointer mismatch")
    supplies = {}
    for fields in _rows(files["sup"]):
        if len(fields) != 3:
            raise ValueError("Expected node / commodity / supply")
        node, commodity, amount = int(fields[0]), int(fields[1]), float(fields[2])
        if not math.isfinite(amount) or not 1 <= node <= n:
            raise ValueError("Invalid Mnetgen supply")
        supplies.setdefault(commodity, []).append((node, amount))
    if set(supplies) != set(range(1, k + 1)):
        raise ValueError("Mnetgen commodity IDs / count mismatch")
    commodities = []
    for identifier, values in sorted(supplies.items()):
        sources = [(node, value) for node, value in values if value > 0]
        sinks = [(node, value) for node, value in values if value < 0]
        if (len(values) != 2 or len(sources) != 1 or len(sinks) != 1
                or not math.isclose(sources[0][1], -sinks[0][1], rel_tol=1e-12, abs_tol=1e-8)):
            raise ValueError("Only balanced single-source/single-sink commodities are supported")
        commodities.append(dict(id=identifier, origin=sources[0][0], destination=sinks[0][0],
                                demand=sources[0][1]))
    checksums = {ext: hashlib.sha256(raw).hexdigest() for ext, raw in files.items()}
    return _instance(list(range(1, n + 1)), arcs, commodities, {
        "benchmark": f"{prefix} (published linear-flow special case)", "instance_name": name,
        "source_url": (FNSS_URL if prefix == "waxman" else MNETGEN_FORMAT_SOURCE),
        "source_files_sha256": checksums,
        "source_sha256": hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest(),
        "model": "directed_splittable_linear_multicommodity",
        "comparison_group": "linear_flow_special_case",
        "adaptation": "lossless_homogeneous_mnetgen_with_zero_fixed_cost",
        "format_source_url": MNETGEN_FORMAT_SOURCE,
        "warning": "No fixed opening costs in source; report separately from fixed-charge SND.",
    })


def load_fnss_benchmark(archive: str | Path, name: str) -> dict:
    """Backward-compatible reader for the published Waxman/FNSS bundle."""
    return load_mnetgen_benchmark(archive, name, prefix="waxman")


def load_benchmark(archive: str | Path, name: str, *, kind: str) -> dict:
    """Dispatch a local published archive; never generate replacement instances."""
    loaders = {"canad": load_canad_benchmark, "holmberg": load_holmberg_benchmark,
               "fnss": load_fnss_benchmark, "mnetgen": load_mnetgen_benchmark}
    if kind not in loaders:
        raise ValueError(f"Unknown benchmark kind: {kind}")
    return loaders[kind](archive, name)


def capacity_infeasibility_certificates(data: dict) -> list[dict]:
    """Cheap necessary capacity conditions, NOT a complete feasibility solver.

    Summing all commodity balances at a node gives net_supply. Its outgoing
    capacity must be >= net_supply, and incoming capacity >= -net_supply.
    A violation proves infeasibility even with every arc open. No violation does
    not prove feasibility; general cuts and multicommodity congestion remain.
    Complexity O(nodes + arcs + commodities), with no LP/MIP or network changes.
    """
    outgoing = dict.fromkeys(data["nodes"], 0.0)
    incoming = dict.fromkeys(data["nodes"], 0.0)
    net = dict.fromkeys(data["nodes"], 0.0)
    for arc in data["arcs"]:
        outgoing[arc["tail"]] += float(arc["capacity"])
        incoming[arc["head"]] += float(arc["capacity"])
    for commodity in data["commodities"]:
        net[commodity["origin"]] += float(commodity["demand"])
        net[commodity["destination"]] -= float(commodity["demand"])
    certificates = []
    for node in data["nodes"]:
        for direction, required, available in (("outgoing", net[node], outgoing[node]),
                                               ("incoming", -net[node], incoming[node])):
            tolerance = 1e-7 * max(1.0, abs(required), abs(available))
            if required > available + tolerance:
                certificates.append({"type": "singleton_capacity_cut", "node": node,
                                     "direction": direction, "required_net_flow": required,
                                     "available_capacity": available,
                                     "deficit": required - available})
    return certificates
