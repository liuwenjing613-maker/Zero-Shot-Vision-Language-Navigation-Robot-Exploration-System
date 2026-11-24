"""Assign against the previous frame's records before creating any new records."""
import math

ALIASES = {"armchair": "chair", "couch": "sofa", "desk": "table"}
UNKNOWN = {"", "unknown", "unclear", "n/a", "unspecified"}


def category(value):
    value = value.casefold().strip()
    return ALIASES.get(value, value)


def attributes(raw):
    return {
        k.casefold().strip(): v.casefold().strip()
        for k, v in raw.items()
        if v.casefold().strip() not in UNKNOWN
    }


def score(obs, obj):
    distance = math.dist(obs.map_position_xz, obj.map_position_xz)
    radius = min(
        1.2,
        max(0.35, 0.28 + 2 * math.hypot(obs.position_sigma_m, obj.position_sigma_m)),
    )
    if distance > radius:
        return 0.0
    cats = obj.category_scores
    semantic = cats.get(category(obs.category_raw), 0.0) / max(cats.values())
    matches = []
    for key, value in attributes(obs.attributes_raw).items():
        if key in obj.attribute_scores:
            votes = obj.attribute_scores[key]
            matches.append(votes.get(value, 0.0) / max(votes.values()))
    attr = sum(matches) / len(matches) if matches else 0.5
    return 0.65 * (1 - distance / radius) + 0.20 * semantic + 0.15 * attr


def associate(observations, records):
    proposals = []
    for obs in observations:
        if obs.map_position_xz is None:
            obs.association_status = "UNLOCALIZED"
            continue
        ranked = sorted(
            ((score(obs, obj), key) for key, obj in records.items()), reverse=True
        )
        obs.candidates = [(key, value) for value, key in ranked if value >= 0.52]
        if not obs.candidates:
            obs.association_status = "NEW"
            continue
        best, value = obs.candidates[0]
        if len(ranked) > 1 and value - ranked[1][0] < 0.08:
            obs.association_status = "AMBIGUOUS"
            continue
        proposals.append((value, obs, best))
    used = set()
    for _, obs, key in sorted(proposals, key=lambda item: -item[0]):
        if key in used:
            obs.association_status = "AMBIGUOUS"
        else:
            obs.association_status = "MATCHED"
            obs.object_id = key
            used.add(key)
