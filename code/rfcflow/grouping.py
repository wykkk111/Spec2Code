import json
from dataclasses import dataclass
from typing import Dict, List

from rfc import SRRecord


@dataclass(frozen=True)
class GroupAssignment:
    group: str
    subgroup: str


def load_group_info(path: str) -> Dict[int, GroupAssignment]:
    """
    Load predefined group_info.json into sr_id -> (group, subgroup).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, GroupAssignment] = {}
    for group, sub in data.items():
        if not isinstance(sub, dict):
            continue
        for subgroup, ids in sub.items():
            if not isinstance(ids, list):
                continue
            for sr_id in ids:
                try:
                    out[int(sr_id)] = GroupAssignment(str(group), str(subgroup))
                except Exception:
                    continue
    return out


def apply_grouping_to_srs(srs: List[SRRecord], mapping: Dict[int, GroupAssignment]) -> List[SRRecord]:
    for sr in srs:
        g = mapping.get(int(sr.sr_id))
        if g:
            sr.group = g.group
            sr.subgroup = g.subgroup
    return srs




