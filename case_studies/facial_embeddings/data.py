"""Shared data loading for the OT-CCv2 pipeline."""

import numpy as np
import json
from collections import defaultdict, Counter

EMBEDDINGS_PATH = "/home/nina.dahora/ccv2/ccv2-audit-kit/outputs/embeddings/all_embeddings.npz"
ANNOTATIONS_PATH = "/home/nina.dahora/dataset/casual_conversations_v2/annotations/CasualConversationsV2.json"


def load_subjects():
    """
    Load CCv2 embeddings and metadata, grouped by subject.

    Returns:
        subjects: dict {sid: {"embeddings": array, "country": ..., "monk": ..., ...}}
        measures: dict {sid: (embeddings, uniform_weights)}
    """
    emb_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    paths, embeddings = emb_data["paths"], emb_data["embeddings"]

    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    meta = {}
    for ann in annotations:
        sid = str(ann["subject_id"]).zfill(4)
        if sid not in meta:
            meta[sid] = {
                "country": ann["geo_location"]["country"],
                "state": ann["geo_location"].get("state_region", "unknown"),
                "native_language": ann["native_language"],
                "monk": ann["monk_skin_tone"]["scale"],
            }

    emb_by_subject = defaultdict(list)
    for path, emb in zip(paths, embeddings):
        sid = str(path).split("/")[2]
        emb_by_subject[sid].append(emb)

    valid = set(emb_by_subject) & set(meta)
    subjects, measures = {}, {}
    for sid in valid:
        embs = np.array(emb_by_subject[sid])
        subjects[sid] = {"embeddings": embs, **meta[sid]}
        measures[sid] = (embs, np.ones(len(embs)) / len(embs))

    countries = Counter(s["country"] for s in subjects.values())
    print(f"{len(subjects)} subjects: {dict(countries.most_common())}")
    return subjects, measures
