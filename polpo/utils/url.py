import requests

import polpo.concurrent as pconcurrent


def is_link_ok(url):
    return requests.get(
        url,
        allow_redirects=True,
        headers={"User-Agent": "link-checker"},
    ).ok


def are_links_ok(urls, workers=16):
    return pconcurrent.thread_map(is_link_ok, urls, workers=workers)


__all__ = ["is_link_ok", "are_links_ok"]
