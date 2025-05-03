import os


def dedupe_by_id(items, key="id"):
    """
    Remove duplicates from a list of dictionaries based on a unique key.
    Preserves the original order.
    :param items: List[dict] – input records
    :param key: str – the dictionary key to deduplicate on
    :return: List[dict] – deduplicated records
    """
    seen = set()
    result = []
    for item in items:
        identifier = item.get(key)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result


def ensure_dir(path):
    """
    Ensure that the directory for the given file or folder path exists.
    If a file path is provided, it creates its parent directory.
    :param path: str – file or directory path
    """
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
