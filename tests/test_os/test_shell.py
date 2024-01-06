import pytest
from pybackpack.os.shell import get_files, find


# Using pytest temporary directory which automatically cleans it up
@pytest.fixture(scope="module")
def test_dir(tmp_path_factory):
    # Create a temporary directory for this module
    base_temp = tmp_path_factory.mktemp("test")
    dir1 = base_temp / "dir1"
    dir1.mkdir(parents=True, exist_ok=True)

    (dir1 / "file1.yml").write_text("content")
    (dir1 / "file2.yaml").write_text("content")
    (dir1 / "file1.dev.yml").write_text("content")
    (dir1 / "file2.dev.yaml").write_text("content")
    (dir1 / "file3.txt").write_text("content")
    (dir1 / "file4.py").write_text("content")
    (dir1 / "file5.yamld").write_text("content")

    # Add a subdirectory
    dir1_sub1 = dir1 / "dir1_sub1"
    dir1_sub1.mkdir(parents=True, exist_ok=True)

    (dir1_sub1 / "file1.txt").write_text("content")
    (dir1_sub1 / "file2.txt").write_text("content")

    return base_temp


def test_get_files_all_files(test_dir):
    # Get all the files
    files = get_files(test_dir)
    assert len(files) == 9

    # Get all files from an unknown directory
    files = get_files("unknown")
    assert not files


def test_get_files(test_dir):
    base_dir = test_dir / "dir1"

    # Get all the yaml files
    files = get_files(base_dir, names=[r".*\.ya?ml$"])
    assert len(files) == 4
    assert {"file3.txt", "file4.py"} not in {file.name for file in files}

    # Get all the files except txt and py files
    files = get_files(base_dir, exclude_names=[r".*\.txt", r".*\.py"])
    assert len(files) == 5
    assert {"file3.txt", "file4.py"} not in {file.name for file in files}

    # Get all the yaml files except the ones with dev in the name
    files = get_files(
        base_dir,
        names=[r".*\.ya?ml$"],
        exclude_names=[r".*dev.*"],
    )
    assert len(files) == 2
    assert {"file1.dev.yml", "file2.dev.yaml"} not in {
        file.name for file in files
    }

    # Finding no files with the given patterns
    files = get_files(base_dir, names=[r".*\.cpp"])
    assert len(files) == 0

    # All *.txt files in all directories
    files = get_files(base_dir, names=[r".*\.txt"])
    assert len(files) == 3

    # Find a particular file
    file_name = "file3"
    files = get_files(base_dir, names=[rf"{file_name}\.txt"])
    assert len(files) == 1


def test_find(test_dir):
    # Get all yaml files
    files = find(test_dir, names=["*.yaml", "*.yml"])
    assert len(files) == 4
    assert {"file3.txt", "file4.py"} not in set(files)

    # Get all the yaml files using regex
    files = find(test_dir, names=[r".*\.ya?ml$"], use_regex=True)
    assert len(files) == 4
    assert {"file3.txt", "file4.py"} not in set(files)

    # Get all the files except txt and py files
    files = find(test_dir, types=["f"], exclude_names=["*.txt", "*.py"])
    assert len(files) == 5
    assert {"file3.txt", "file4.py"} not in set(files)

    # Get all the yaml files except the ones with dev in the name
    files = find(
        test_dir,
        names=[r".*\.ya?ml$"],
        exclude_names=[r".*dev.ya?ml$"],
        use_regex=True,
    )
    assert len(files) == 2
    assert {"file1.dev.yml", "file2.dev.yaml"} not in set(files)

    # Finding no files with the given patterns
    files = find(test_dir, names=["*.cpp"])
    assert len(files) == 0

    # All *.txt files in all directories
    files = find(test_dir, names=["*.txt"])
    assert len(files) == 3

    # Find a particular file
    file_name = "file3"
    files = find(test_dir, names=[rf"{file_name}\.txt"])
    assert len(files) == 1
