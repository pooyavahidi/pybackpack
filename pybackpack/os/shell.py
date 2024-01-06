import re
from pathlib import Path
from typing import List, Optional
from pybackpack.os import run_shell_command


def get_files(
    directory: str,
    names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[Path]:
    """Returns a list of files in the dir_path directory.

    Args:
        directory: The directory to search for files.
        names: A list of regular expressions to match the filenames.
        by default, all files are included i.e. include_patterns=[".*"].
        exclude_names: A list of regular expressions to exclude the
            filenames.
        recursive: If True, search recursively in the dir_path directory.
        default is True.
    """

    files = []

    # If names is None, then include all files
    if names is None:
        names = [r".*"]

    # Get files based on the value of recursive
    if recursive:
        paths = list(Path(directory).rglob("*"))
    else:
        paths = list(Path(directory).glob("*"))

    for path in paths:
        # Skip directories or other non-files
        if not path.is_file():
            continue

        should_include = any(
            re.search(pattern, path.name) for pattern in names
        )
        if not should_include:
            continue

        if exclude_names:
            should_exclude = any(
                re.search(pattern, path.name) for pattern in exclude_names
            )
            if should_exclude:
                continue

        files.append(path)

    return files


def find(
    directory: str,
    names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
    use_regex: bool = False,
) -> List[str]:
    """Wrapper around the find command in Unix-like systems.
    Args:
        directory (str): The directory to search.
        name (list): A list of object names to search for.
        exclude_name (list): A list of object names to exclude.
        types (list): A list of object types to search for.
            The common types are: f (file), d (directory), l (symbolic link).
            For the complete list of possible options see `find` command help.
        use_regex (bool): Use regex for file name search. Default is False.
            if False, use glob patterns for names.
    Returns:
        list: A list of objects found.
    """

    # A simple AST-like template for constructing the find command
    ast = {
        "base": ["find", "{}"],
        "name": ["-name", "{}"] if not use_regex else ["-regex", "{}"],
        "type": ["-type", "{}"],
        "or": "-o",
        "not": "!",
        "open_paren": "(",
        "close_paren": ")",
    }

    cmd = ast["base"].copy()
    cmd[1] = cmd[1].format(directory)

    # Handle object types
    if types:
        cmd.append(ast["open_paren"])
        for otype in types:
            cmd.extend(ast["type"])
            cmd[-1] = cmd[-1].format(otype)
            cmd.append(ast["or"])
        cmd.pop()  # remove the last '-o'
        cmd.append(ast["close_paren"])

    # Handle object names
    if names:
        for name in names:
            cmd.extend(ast["name"])
            cmd[-1] = cmd[-1].format(name)
            cmd.append(ast["or"])
        cmd.pop()  # remove the last '-o'

    # Handle excluded names
    if exclude_names:
        cmd.append(ast["not"])
        cmd.append(ast["open_paren"])
        for ex in exclude_names:
            cmd.extend(ast["name"])
            cmd[-1] = cmd[-1].format(ex)
            cmd.append(ast["or"])
        cmd.pop()  # remove the last '-o'
        cmd.append(ast["close_paren"])

    return run_shell_command(cmd)
