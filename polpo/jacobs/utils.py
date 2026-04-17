MATERNAL_IDS = {
    "01",
    "1001B",
    "1004B",
    "1009B",
    "2004B",
    "3004B",
}


def get_subject_ids(
    include_pilot=True, include_male=True, include_control=True, sort=False
):
    ids = MATERNAL_IDS.copy()

    if not include_pilot:
        ids.remove("01")

    for id_ in ids.copy():
        if not include_male and id_.startswith("2"):
            ids.remove(id_)

        if not include_control and id_.startswith("3"):
            ids.remove(id_)

    if sort:
        ids = sorted(ids)

    return ids
