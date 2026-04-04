MATERNAL_IDS = {"01", "1001B", "1004B", "2004B", "1009B"}


def get_subject_ids(include_pilot=True, include_male=True, sort=False):
    ids = MATERNAL_IDS.copy()

    if not include_pilot:
        ids.remove("01")

    if not include_male:
        for id_ in ids.copy():
            if id_.startswith("2"):
                ids.remove(id_)

    if sort:
        ids = sorted(ids)

    return ids
