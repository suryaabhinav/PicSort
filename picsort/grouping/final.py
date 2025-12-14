from typing import Counter

import pandas as pd

from api.logging_config import log


def compute_id_image_counts(df_stage_c: pd.DataFrame) -> pd.DataFrame:
    """Compute the number of images each face ID appears in."""

    id_image_counts = Counter()
    for _, row in df_stage_c.iterrows():
        labels = row.get("_face_id_labels", [])
        if not isinstance(labels, (list, tuple)):
            continue
        idx_in_image = {l for l in labels if isinstance(l, int) and l >= 0}
        for pid in idx_in_image:
            id_image_counts[pid] += 1
    return id_image_counts


def final_group_name(row: pd.Series, id_image_counts: Counter) -> str:
    """Generate a final group name based on face ID counts."""

    if row["focus_label"] != "subject_in_focus":
        return "00_LowQuality"

    person_count = int(row.get("person_count", 0))
    faces = int(row.get("num_faces_found", 0))

    if person_count == 0:
        scene_group = int(row.get("scene_group", -100))
        if scene_group >= 0:
            return f"20_Landscape/Scene_{scene_group}"
        if scene_group == -1:
            return f"20_Landscape/99_Unique_Noise"
        return "30_Outliers/99_Unclassified_Sharp"

    if 2 <= faces <= 5:
        return "10_People/Group"

    if faces > 5:
        return "10_People/98_Large_Group"

    if faces == 0:
        return "10_People/97_Silhouette"

    labels = row.get("_face_id_labels", [])
    valid_ids = sorted(set([l for l in labels if isinstance(l, int) and l >= 0]))

    def _person_or_singleton(pid: int) -> str:
        count = int(id_image_counts.get(pid, 0))
        if count >= 2:
            return f"10_People/Person_{pid}"
        else:
            return "10_People/96_Singleton"

    if len(valid_ids) == 1:
        return _person_or_singleton(valid_ids[0])

    if len(valid_ids) > 1:
        counts = Counter([l for l in labels if isinstance(l, int) and l >= 0])
        if counts:
            top_id, top_cnt = counts.most_common(1)[0]
            if list(counts.values()).count(top_cnt) == 1:
                return _person_or_singleton(top_id)

        return "10_People/99_Unidentified_Noise"

    return "10_People/99_Unidentified_Noise"


def grouping(df_stage_c: pd.DataFrame) -> pd.DataFrame:
    """Apply final grouping to images in DataFrame

    Args:
        df_stage_c (pd.DataFrame): DataFrame from stage C

    Returns:
        pd.DataFrame: DataFrame with final grouping
    """

    id_counts = compute_id_image_counts(df_stage_c)
    df_final = df_stage_c.copy()
    df_final["final_group_name"] = df_final.apply(
        lambda row: final_group_name(row, id_counts), axis=1
    )
    log.info("[INFO] Final Grouping Done")
    return df_final
