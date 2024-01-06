import pytest
import pytest
from pybackpack.os import ProcessCommand, run_shell_command
from pybackpack.commands import (
    PipeCommand,
    SequentialCommand,
    run_command,
)

# pylint: disable=redefined-outer-name


def test_shell_command():
    cmd = ProcessCommand(["echo", "Hello"])
    res = cmd.run()
    assert res.output == "Hello\n"
    assert res.metadata.returncode == 0

    # Test with env variables
    cmd = ProcessCommand(["env"], env={"MY_VAR": "test1"})
    res = run_command(cmd)
    assert "MY_VAR=test1" in res.output

    # Test timeout
    cmd = ProcessCommand(["sleep", "2"], timeout=1)
    res = run_command(cmd)
    assert res.output is None
    assert res.succeeded is False
    assert res.metadata["failure_reason"] == "timeout"

    # Test failed command
    cmd = ProcessCommand(["ls", "unknown"])
    res = run_command(cmd)
    assert res.output is None
    assert res.succeeded is False
    assert res.error.returncode == 2
    assert "No such file or directory" in res.error.stderr

    # Test failed command. Command doesn't raise error but returns the failure
    cmd = ProcessCommand(["ls", "unknown"])
    res = run_command(cmd)
    assert res.succeeded is False
    assert res.error.returncode == 2

    # Test failed with command not found
    cmd = ProcessCommand(["unknown"])
    res = run_command(cmd)
    assert res.output is None
    assert res.succeeded is False
    assert res.error.errno == 2
    assert res.metadata["failure_reason"] == "command_not_found"
    assert "No such file or directory" in res.error.strerror
    assert "No such file or directory" in res.error_message

    # Multiline output
    cmd = ProcessCommand(["ls", "/", "-l"])
    res = run_command(cmd)
    assert res.output is not None
    assert len(res.output.splitlines()) > 1


def test_pipe():
    # Pipe commands
    commands = [
        ProcessCommand(["echo", "Hello World"]),
        ProcessCommand(["cut", "-d", " ", "-f", "1"]),
        ProcessCommand(["awk", "{print $1}"]),
    ]
    pipe = PipeCommand(commands)
    res = run_command(pipe)
    assert res.succeeded is True
    assert res.output == "Hello\n"
    assert res.results[0].metadata.returncode == 0
    assert res.results[0].metadata.stdout == "Hello World\n"
    assert res.results[1].metadata.returncode == 0
    assert res.results[2].metadata.returncode == 0

    # Test with a failing command in the middle
    commands = [
        ProcessCommand(["echo", "Hello World"]),
        ProcessCommand(["cut", "-d", " ", "-f", "1"]),
        ProcessCommand(["unknown"]),
        ProcessCommand(["awk", "{print $1}"]),
    ]
    pipe = PipeCommand(commands)
    res = run_command(pipe)
    assert res.output is None
    assert res.succeeded is False
    assert len(res.results) == 3
    assert res.results[0].succeeded is True
    assert res.results[0].output == "Hello World\n"
    assert res.results[1].succeeded is True
    assert res.results[1].output == "Hello\n"
    assert res.results[2].succeeded is False
    assert res.results[2].output is None


def test_sequential():
    # Simulate && operator in shell
    commands = [
        ProcessCommand(["echo", "Hello"]),
        ProcessCommand(["echo", "World"]),
    ]
    seq = SequentialCommand(commands, collect_results=True)
    res = run_command(seq)
    assert {"Hello\n", "World\n"} == set(res.output)

    # Simulate && with error
    commands = [
        ProcessCommand(["echo", "Hello"]),
        ProcessCommand(["ls", "unknown"]),
        ProcessCommand(["echo", "World"]),
    ]
    seq = SequentialCommand(commands)
    res = run_command(seq)
    assert res.output == ["Hello\n", None]
    assert res.succeeded is False

    # The first command was successful
    assert res.results[0].metadata.returncode == 0
    assert res.results[0].metadata.stdout == "Hello\n"
    # The command which failed
    assert res.results[1].succeeded is False
    assert res.results[1].error.returncode == 2
    assert "No such file or directory" in res.results[1].error.stderr

    # Simulate || operator in shell
    commands = [
        ProcessCommand(["echo", "Hello"]),
        ProcessCommand(["ls", "unknown"]),
        ProcessCommand(["echo", "World"]),
    ]
    seq = SequentialCommand(commands, operator="||")
    res = run_command(seq)
    assert res.output == ["Hello\n"]

    # Simulatte ; operator in shell
    commands = [
        ProcessCommand(["echo", "Hello"]),
        ProcessCommand(["ls", "unknown"]),
        ProcessCommand(["echo", "World"]),
    ]
    seq = SequentialCommand(commands, operator=None, collect_results=True)
    res = run_command(seq)
    assert res.output == ["Hello\n", None, "World\n"]

    # Invalid operator
    with pytest.raises(ValueError):
        SequentialCommand(commands, operator="invalid").run()


def test_sequential_combined(in_vscode_launch):
    # Sequential combined
    cmd1 = SequentialCommand(
        [
            ProcessCommand(["echo", "1"]),
            ProcessCommand(["python3", "-c", "import os; print(os.getpid())"]),
        ]
    )
    cmd2 = SequentialCommand(
        [
            ProcessCommand(["echo", "2"]),
            ProcessCommand(["python3", "-c", "import os; print(os.getpid())"]),
        ]
    )
    cmd3 = SequentialCommand(
        [
            ProcessCommand(["echo", "3"]),
            ProcessCommand(["python3", "-c", "import os; print(os.getpid())"]),
        ]
    )

    # Run the commands in sequence
    seq = SequentialCommand([cmd1, cmd2, cmd3], collect_results=True)
    res = run_command(seq)
    assert res.succeeded is True

    if in_vscode_launch:
        assert len(res.output) == 3
    else:
        assert res.output[0] != res.output[1] != res.output[2]


def test_run_shel_command():
    # Test multi-lines output
    cmd = ["ls", "/", "-l"]
    res = run_shell_command(cmd)
    assert len(res) > 1

    # Test failure
    cmd = ["ls", "unknown"]
    with pytest.raises(Exception) as ex_info:
        run_shell_command(cmd)
    assert "No such file or directory" in str(ex_info.value.stderr)
