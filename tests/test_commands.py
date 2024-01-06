import os
import asyncio
import random
import pytest
from pybackpack.commands import (
    Command,
    AsyncCommand,
    CommandResult,
    PipeCommand,
    SequentialCommand,
    MultiProcessCommand,
    AsyncAdapterCommand,
    SyncAdapterCommand,
    AsyncConcurrentCommand,
    run_command,
    async_run_command,
)

# pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name


class EchoCommand(Command):
    def __init__(self, name=None) -> None:
        self.name = name

    def run(self, input_data=None):
        if not input_data:
            input_data = ""
        return CommandResult(f"{input_data}{self.name}")


class FailingCommand(Command):
    def __init__(self, raise_error=False) -> None:
        self.error_message = "Error from ErrorCommand"
        self.raise_error = raise_error

    def run(self, input_data=None):
        if self.raise_error:
            raise SystemError(self.error_message)

        return CommandResult(
            output=None,
            succeeded=False,
            error_message=self.error_message,
        )


class ProcessInfoCommand(Command):
    def run(self, input_data=None):
        return CommandResult(output=os.getpid())


class AsyncTestCommand(AsyncCommand):
    def __init__(
        self,
        shared_counter=None,
        concurrency_limit=2,
        name="A",
        activity_logs=None,
        work_simulation_delay=0.001,
        raise_error=False,
    ):
        self.shared_counter = shared_counter
        if self.shared_counter is None:
            self.shared_counter = {"current": 0}

        self.concurrency_limit = concurrency_limit
        self.name = name
        self.activity_logs = activity_logs
        self.work_simulation_delay = work_simulation_delay
        self.raise_error = raise_error

    async def async_run(self, input_data=None):
        if self.activity_logs is not None:
            self.activity_logs.append(self.name)

        self.shared_counter["current"] += 1
        # Check if the current concurrency exceeds the limit
        if self.shared_counter["current"] > self.concurrency_limit:
            raise RuntimeError("Concurrency limit exceeded")

        # Short delay to simulate work and allow concurrency
        await asyncio.sleep(self.work_simulation_delay)

        # Job is done, so decrement the current concurrency
        self.shared_counter["current"] -= 1

        # To simulate error, raise error if `raise_error` is True
        if self.raise_error:
            raise RuntimeError("Error from AsyncTestCommand")

        if input_data is None:
            input_data = ""
        return CommandResult(output=f"{input_data}{self.name}")


def test_single_command():
    cmd = EchoCommand("A")
    res = run_command(cmd)
    assert res.output == "A"
    assert res.succeeded is True

    # Command with error
    cmd = FailingCommand(raise_error=True)
    res = run_command(cmd)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, SystemError)
    assert res.error_message == "Error from ErrorCommand"

    # Command with error which doesn't raise error
    cmd = FailingCommand(raise_error=False)
    res = run_command(cmd)
    assert res.output is None
    assert res.succeeded is False
    assert res.error_message == "Error from ErrorCommand"

    # Test always raising error command
    cmd = FailingCommand(raise_error=True)
    res = run_command(cmd)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, SystemError)

    # Pass None as input
    cmd = EchoCommand("A")
    res = run_command(cmd, input_data=None)
    assert res.output == "A"
    assert res.succeeded is True

    # Pass empty string as input
    cmd = EchoCommand("A")
    res = run_command(cmd, input_data="")
    assert res.output == "A"
    assert res.succeeded is True


@pytest.mark.asyncio
async def test_single_command_async():
    # Command without error
    cmd = AsyncTestCommand(name="A")
    res = await async_run_command(cmd)
    assert res.output == "A"
    assert res.succeeded is True

    # Command which raises error
    cmd = AsyncTestCommand(raise_error=True)
    res = await async_run_command(cmd)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, RuntimeError)
    assert "Error" in res.error_message


def test_pipe():
    # Test pipe with simple commands, no errors
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    pipe = PipeCommand(commands=commands)
    res = run_command(pipe)
    assert res.output == "ABC"

    # Test pipe with initial input to the pipe
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    pipe = PipeCommand(commands)
    res = run_command(pipe, input_data="D")
    assert res.output == "DABC"
    assert len(res.results) == 3
    assert res.results[0].output == "DA"
    assert res.results[0].succeeded is True
    assert res.results[1].output == "DAB"
    assert res.results[2].output == "DABC"

    # Test pipe with command with error which doesn't raise error
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
    ]
    pipe = PipeCommand(commands)
    res = run_command(pipe)
    assert res.output is None
    assert res.succeeded is False
    assert res.error_message == "Error from ErrorCommand"
    # The error is None as Command doesn't raise error
    assert res.error is None
    assert res.results[0].output == "A"

    # Test pipe with command with error which raises error
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=True),
        EchoCommand("C"),
    ]
    pipe = PipeCommand(commands)
    res = run_command(pipe)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, SystemError)

    # Test pipe with None commands
    with pytest.raises(ValueError):
        pipe = PipeCommand(commands=None)

    # Test pipe with empty commands
    with pytest.raises(ValueError):
        pipe = PipeCommand(commands=[])


def test_pipe_collect_results():
    # Test without collecting results
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
    ]
    pipe = PipeCommand(commands, collect_results=False)
    res = run_command(pipe)
    assert res.output is None
    assert res.succeeded is False
    assert res.error_message == "Error from ErrorCommand"
    assert res.results == []


@pytest.mark.asyncio
async def test_pipe_async():
    # Test with no errors
    activity_logs = []
    commands = [
        AsyncTestCommand(name="A", activity_logs=activity_logs),
        AsyncTestCommand(name="B", activity_logs=activity_logs),
        AsyncTestCommand(name="C", activity_logs=activity_logs),
    ]
    pipe = PipeCommand(commands)
    res = await async_run_command(pipe)
    assert res.succeeded is True
    assert res.output == "ABC"
    assert activity_logs == ["A", "B", "C"]

    # Test pipe with initial input to the pipe
    commands = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(name="B"),
        AsyncTestCommand(name="C"),
    ]
    pipe = PipeCommand(commands)
    res = await async_run_command(pipe, input_data="D")
    assert res.output == "DABC"
    assert res.results[0].output == "DA"
    assert res.results[0].succeeded is True
    assert res.results[1].output == "DAB"
    assert res.results[2].output == "DABC"

    # Test without collecting results
    commands = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(raise_error=True),
        AsyncTestCommand(name="C"),
    ]
    pipe = PipeCommand(commands, collect_results=False)
    res = await async_run_command(pipe)
    assert res.output is None
    assert res.succeeded is False
    assert "Error" in res.error_message

    # Test with errors
    activity_logs = []
    delay = 0.001
    commands = [
        AsyncTestCommand(
            name="A", activity_logs=activity_logs, work_simulation_delay=delay
        ),
        AsyncTestCommand(
            name="B", activity_logs=activity_logs, work_simulation_delay=delay
        ),
        AsyncTestCommand(
            name="Error",
            activity_logs=activity_logs,
            raise_error=True,
            work_simulation_delay=delay,
        ),
        AsyncTestCommand(
            name="C", activity_logs=activity_logs, work_simulation_delay=delay
        ),
    ]
    pipe = PipeCommand(commands)
    res = await async_run_command(pipe)
    assert res.succeeded is False
    assert activity_logs == ["A", "B", "Error"]
    assert res.results[0].output == "A"
    assert res.results[1].output == "AB"
    assert res.results[2].output is None
    assert res.output is None


def test_sequential_and_operator():
    # Simulate && in unix-like systems
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands)
    res = run_command(seq)
    assert res.succeeded is True
    assert res.output == ["A", "B", "C"]
    # It can also be achieved using `results` attribute as both are the same.
    # `results` holds all the additional information about the command
    # execution such as error, output, etc.
    outputs = [r.output for r in res.results]
    assert outputs == ["A", "B", "C"]

    # Simulate && with error
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands)
    res = run_command(seq)
    assert res.output == ["A", None]
    assert res.succeeded is False

    # Simulate && with error
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands)
    res = run_command(seq)
    assert res.output == ["A", None]
    assert res.succeeded is False

    # Test with no commands
    with pytest.raises(ValueError):
        seq = SequentialCommand(commands=None)

    # Test with empty commands
    with pytest.raises(ValueError):
        seq = SequentialCommand(commands=[])


def test_sequential_or_operator():
    # Simulate || in unix-like system
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands, operator="||")
    res = run_command(seq)
    assert res.output == ["A"]

    # Simulate || with error early
    commands = [
        FailingCommand(raise_error=False),
        FailingCommand(raise_error=True),
        EchoCommand("A"),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands, operator="||")
    res = run_command(seq)
    assert set(res.output) >= {"A"}
    assert res.succeeded is True
    assert res.results[0].output is None
    assert isinstance(res.results[1].error, SystemError)
    assert res.results[2].output == "A"

    # Simulate || without error
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands, operator="||")
    res = run_command(seq)
    assert res.output == ["A"]
    assert res.succeeded is True

    # Simulate || with collecting outputs
    commands = [
        FailingCommand(raise_error=False),
        EchoCommand("A"),
        EchoCommand("C"),
    ]
    seq = SequentialCommand(commands, operator="||")
    res = run_command(seq)
    assert res.output == [None, "A"]
    assert res.succeeded is True
    assert res.results[0].output is None
    assert res.results[1].output == "A"
    assert len(res.results) == 2


def test_sequential_no_operator():
    # Simulate ; in unix-like system
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
        FailingCommand(raise_error=True),
        EchoCommand("B"),
        FailingCommand(raise_error=True),
    ]
    seq = SequentialCommand(commands, operator=None)
    res = run_command(seq)
    assert [res.succeeded for res in res.results] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert res.output == ["A", None, "C", None, "B", None]
    assert res.results[1].succeeded is False
    assert res.results[1].error is None
    assert isinstance(res.results[3].error, SystemError)
    assert res.results[5].succeeded is False
    assert isinstance(res.results[5].error, SystemError)


def test_sequential_combined():
    # (A && B) && (C && D) = A && B && C && D
    seq1 = SequentialCommand([EchoCommand("A"), EchoCommand("B")])
    seq2 = SequentialCommand([EchoCommand("C"), EchoCommand("D")])
    seq = SequentialCommand([seq1, seq2])
    res = run_command(seq)
    assert res.output == [["A", "B"], ["C", "D"]]
    assert res.succeeded is True
    assert res.results[0].output == ["A", "B"]
    # Directly accessing the output of the first command in seq2
    assert res.results[1].results[0].output == "C"
    # Directly access the outputs using the `output` attribute
    assert res.output[1][0] == "C"

    # (A && B) || (C && D)
    seq1 = SequentialCommand([EchoCommand("A"), EchoCommand("B")])
    seq2 = SequentialCommand([EchoCommand("C"), EchoCommand("D")])
    seq = SequentialCommand([seq1, seq2], operator="||")
    res = run_command(seq)
    assert res.output == [["A", "B"]]
    assert res.succeeded is True
    assert res.results[0].output == ["A", "B"]
    assert len(res.results) == 1

    # Combined using process info
    seq = SequentialCommand(
        [ProcessInfoCommand(), ProcessInfoCommand(), ProcessInfoCommand()],
    )
    res = run_command(seq)
    assert res.succeeded is True
    # Process Ids should be identical since both are run in the same process
    assert res.output[0] == res.output[1] == res.output[2]


@pytest.mark.asyncio
async def test_sequential_async():
    # Simulate && with error
    # A && Error && C = A
    activity_logs = []
    delay = 0.001
    commands = [
        AsyncTestCommand(
            name="A", work_simulation_delay=delay, activity_logs=activity_logs
        ),
        AsyncTestCommand(
            name="Error",
            raise_error=True,
            work_simulation_delay=delay,
            activity_logs=activity_logs,
        ),
        AsyncTestCommand(
            name="C", work_simulation_delay=delay, activity_logs=activity_logs
        ),
    ]
    seq = SequentialCommand(commands)
    res = await async_run_command(seq)
    assert res.output == ["A", None]
    assert res.succeeded is False
    assert activity_logs == ["A", "Error"]

    # Simulate ; in unix-like system
    activity_logs = []
    delay = 0.001
    commands = [
        AsyncTestCommand(
            name="A", work_simulation_delay=delay, activity_logs=activity_logs
        ),
        AsyncTestCommand(
            name="Error1",
            raise_error=True,
            work_simulation_delay=delay,
            activity_logs=activity_logs,
        ),
        AsyncTestCommand(
            name="C", work_simulation_delay=delay, activity_logs=activity_logs
        ),
        AsyncTestCommand(
            name="Error2",
            raise_error=True,
            work_simulation_delay=delay,
            activity_logs=activity_logs,
        ),
        AsyncTestCommand(
            name="B", work_simulation_delay=delay, activity_logs=activity_logs
        ),
        AsyncTestCommand(
            name="Error3",
            raise_error=True,
            work_simulation_delay=delay,
            activity_logs=activity_logs,
        ),
    ]
    seq = SequentialCommand(commands, operator=None)
    res = await async_run_command(seq)
    assert [res.succeeded for res in res.results] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert res.output == ["A", None, "C", None, "B", None]
    assert activity_logs == ["A", "Error1", "C", "Error2", "B", "Error3"]
    assert res.results[1].succeeded is False
    assert res.results[1].error is not None
    assert isinstance(res.results[3].error, RuntimeError)
    assert res.results[5].succeeded is False
    assert isinstance(res.results[5].error, RuntimeError)

    # Test combined sequential commands with different operators
    # (A && B) || (C && D) = A and B
    delay = 0.001
    activity_logs = []
    seq1 = SequentialCommand(
        [
            AsyncTestCommand(
                name="A",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
            AsyncTestCommand(
                name="B",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
        ]
    )
    seq2 = SequentialCommand(
        [
            AsyncTestCommand(
                name="C",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
            AsyncTestCommand(
                name="D",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
        ]
    )
    seq = SequentialCommand([seq1, seq2], operator="||")
    res = await async_run_command(seq)
    assert activity_logs == ["A", "B"]
    assert res.succeeded is True
    assert res.output == [["A", "B"]]

    # Test combined sequential commands
    # (A && Err) || (C && D) = C and D
    activity_logs = []
    seq1 = SequentialCommand(
        [
            AsyncTestCommand(name="A", activity_logs=activity_logs),
            AsyncTestCommand(
                name="Error", raise_error=True, activity_logs=activity_logs
            ),
        ]
    )
    seq2 = SequentialCommand(
        [
            AsyncTestCommand(name="C", activity_logs=activity_logs),
            AsyncTestCommand(name="D", activity_logs=activity_logs),
        ]
    )
    seq = SequentialCommand([seq1, seq2], operator="||")
    res = await async_run_command(seq)
    assert res.succeeded is True
    assert activity_logs == ["A", "Error", "C", "D"]
    assert res.output == [["A", None], ["C", "D"]]
    # Last output
    assert res.output[-1] == ["C", "D"]
    # Last result
    assert res.results[-1].output == ["C", "D"]

    # Test combined sequential commands
    # (Err || B) && (C || D) = B and C
    delay = 0.001
    activity_logs = []
    seq1 = SequentialCommand(
        [
            AsyncTestCommand(
                name="Error",
                raise_error=True,
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
            AsyncTestCommand(
                name="B",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
        ],
        operator="||",
    )
    seq2 = SequentialCommand(
        [
            AsyncTestCommand(
                name="C",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
            AsyncTestCommand(
                name="D",
                activity_logs=activity_logs,
                work_simulation_delay=delay,
            ),
        ],
        operator="||",
    )
    seq = SequentialCommand([seq1, seq2])
    res = await async_run_command(seq)
    assert res.succeeded is True
    assert activity_logs == ["Error", "B", "C"]
    assert res.output == [[None, "B"], ["C"]]


def test_sequential_collect_results():
    seq1 = SequentialCommand(
        [EchoCommand("A"), FailingCommand(raise_error=True)]
    )
    seq2 = SequentialCommand([EchoCommand("C"), EchoCommand("D")])
    seq = SequentialCommand([seq1, seq2], operator="||", collect_results=False)
    res = run_command(seq)
    assert res.succeeded is True
    assert res.output == []
    assert res.results == []

    # Partial collection of results
    seq1 = SequentialCommand(
        [EchoCommand("A"), FailingCommand(raise_error=True)]
    )
    seq2 = SequentialCommand(
        [EchoCommand("C"), EchoCommand("D")], collect_results=False
    )
    seq = SequentialCommand([seq1, seq2], operator="||")
    res = run_command(seq)
    assert res.succeeded is True
    assert res.output == [["A", None], []]
    assert len(res.results) == 2
    assert res.results[0].output == ["A", None]
    assert res.results[1].output == []


def test_parallel():
    # Run Parallel commands without error
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    par = MultiProcessCommand(commands)
    res = run_command(par)
    assert res.output == ["A", "B", "C"]
    assert res.succeeded is True
    assert res.results[0].output == "A"

    # Run Parallel commands with error, raise_error=False
    commands = [
        EchoCommand("A"),
        FailingCommand(raise_error=False),
        EchoCommand("C"),
        FailingCommand(raise_error=True),
        EchoCommand("B"),
        FailingCommand(raise_error=False),
    ]
    par = MultiProcessCommand(commands)
    res = run_command(par)
    assert res.succeeded is True
    assert res.output == ["A", None, "C", None, "B", None]

    # Test Parallel with input data
    commands = [
        EchoCommand("A"),
        EchoCommand("B"),
        EchoCommand("C"),
    ]
    par = MultiProcessCommand(commands)
    res = run_command(par, input_data="D")
    assert res.output == ["DA", "DB", "DC"]
    assert res.succeeded is True


def test_parallel_with_pid(in_vscode_launch):
    # Combined using process info
    seq = MultiProcessCommand(
        [ProcessInfoCommand(), ProcessInfoCommand(), ProcessInfoCommand()],
        pool_size=3,
    )
    res = run_command(seq)
    assert res.succeeded is True

    # Process Ids are different since they are run in different processes.
    # If running in VSCode's debugger, the process Ids will be the same. So,
    # we check for the flag and assert the result accordingly.
    # If running outside the debugger, the process Ids will be different.
    # i.e. using `pytest` command in terminal.
    if in_vscode_launch:
        assert len(res.output) == 3
    else:
        assert res.output[0] != res.output[1] != res.output[2]


@pytest.mark.asyncio
async def test_sync_to_async():
    # Create a sync only command which doesn't support async run.
    sync_cmd = EchoCommand("A")
    # first run the command in a synchronously
    res = run_command(sync_cmd)
    assert res.succeeded is True
    assert res.output == "A"

    # Run the command asynchronously, which should fail as there is no
    # async_run method
    res = await async_run_command(sync_cmd)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, AttributeError)

    # Convert the sync command to async command using Ascynchronizer decorator
    async_cmd = AsyncAdapterCommand(sync_cmd)
    res = await async_run_command(async_cmd)
    assert res.succeeded is True
    assert res.output == "A"


@pytest.mark.asyncio
async def test_async_to_sync():
    # Create an async only command which doesn't support sync run.
    async_cmd = AsyncTestCommand()
    # first run the command in a asynchronously
    res = await async_run_command(async_cmd)
    assert res.succeeded is True
    assert res.output == "A"

    # Run the command synchronously, which should fail as there is no run
    # method
    res = run_command(async_cmd)
    assert res.succeeded is False
    assert res.output is None
    assert isinstance(res.error, AttributeError)

    # Convert the async command to sync command using Synchronizer decorator
    sync_cmd = SyncAdapterCommand(async_cmd)
    res = run_command(sync_cmd)
    assert res.succeeded is True
    assert res.output == "A"


@pytest.mark.asyncio
async def test_async_concurrent():
    # Simple list of commands
    cmds = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(name="B"),
        AsyncTestCommand(name="C"),
    ]
    ac_cmd = AsyncConcurrentCommand(cmds)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert res.output == ["A", "B", "C"]
    assert res.results[2].output == "C"

    # List of commands with command with error
    cmds = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(raise_error=True),
        AsyncTestCommand(name="C"),
    ]
    ac_cmd = AsyncConcurrentCommand(cmds)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert res.output == ["A", None, "C"]
    assert res.results[0].output == "A"
    assert res.results[1].output is None

    # List of commands with command which doesn't support async run
    cmds = [
        AsyncTestCommand(name="A"),
        EchoCommand("B"),
        AsyncTestCommand(name="C"),
    ]
    ac_cmd = AsyncConcurrentCommand(cmds)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert res.output == ["A", None, "C"]
    assert isinstance(res.results[1].error, AttributeError)


@pytest.mark.asyncio
async def test_async_concurrent_semaphore():
    # Run commands within the supported concurrency limit
    shared_counter = {"current": 0}
    supported_concurrency_limit = 2
    commands = [
        AsyncTestCommand(shared_counter, supported_concurrency_limit)
        for _ in range(5)
    ]
    ac_cmd = AsyncConcurrentCommand(commands, concurrency_limit=2)

    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert len(res.output) == 5
    assert all(output == "A" for output in res.output)

    # Run commands with concurrency limit exceeded. It will return results with
    # error
    ac_cmd = AsyncConcurrentCommand(commands, concurrency_limit=0)
    res = await async_run_command(ac_cmd)
    assert any(res.error is not None for res in res.results)


@pytest.mark.asyncio
async def test_async_concurrent_callback():
    # Commands
    cmds = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(name="B"),
        AsyncTestCommand(name="C"),
    ]

    def callback(command, result, **kwargs):
        callback_outputs.append(result.output)

    callback_outputs = []
    ac_cmd = AsyncConcurrentCommand(cmds, callback=callback)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert callback_outputs == ["A", "B", "C"]

    # Commands with error
    cmds = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(raise_error=True),
        AsyncTestCommand(name="C"),
    ]
    callback_outputs = []
    ac_cmd = AsyncConcurrentCommand(cmds, callback=callback)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert callback_outputs == ["A", None, "C"]


def test_async_concurrent_to_sync():
    # Simple list of commands
    cmds = [
        AsyncTestCommand(name="A"),
        AsyncTestCommand(name="B"),
        AsyncTestCommand(name="C"),
    ]
    ac_cmd = AsyncConcurrentCommand(cmds)
    # Convert async command to sync command
    sync_cmd = SyncAdapterCommand(ac_cmd)
    res = run_command(sync_cmd)
    assert res.succeeded is True
    assert res.output == ["A", "B", "C"]
    assert res.results[2].output == "C"

    # Use concurrency limit
    cmds = [AsyncTestCommand(concurrency_limit=2) for _ in range(5)]
    ac_cmd = AsyncConcurrentCommand(cmds, concurrency_limit=2)
    sync_cmd = SyncAdapterCommand(ac_cmd)
    res = run_command(sync_cmd)
    assert res.succeeded is True
    assert len(res.output) == 5
    assert all(output == "A" for output in res.output)


@pytest.mark.asyncio
async def test_async_concurrent_stop_at_failure():
    # Run commands concurrently without stopping at failure
    cmds = [
        AsyncTestCommand(name="A", work_simulation_delay=0.001),
        AsyncTestCommand(name="B", work_simulation_delay=0.002),
        AsyncTestCommand(name="C", work_simulation_delay=0.003),
        AsyncTestCommand(name="D", work_simulation_delay=0.01),
        AsyncTestCommand(raise_error=True, work_simulation_delay=0.006),
        AsyncTestCommand(name="Z", work_simulation_delay=0.001),
    ]
    ac_cmd = AsyncConcurrentCommand(cmds)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is True
    assert res.output == ["A", "B", "C", "D", None, "Z"]

    ac_cmd = AsyncConcurrentCommand(cmds, cancel_on_failure=True)
    res = await async_run_command(ac_cmd)
    assert res.succeeded is False
    assert len(res.results) == 6
    # The task D is a longer running task, so it will be cancelled as soon as
    # the error is raised. Cancelled task still have the output, but it is None
    assert res.output == ["A", "B", "C", None, None, "Z"]
    assert res.results[3].succeeded is False
    assert res.results[3].cancelled is True
