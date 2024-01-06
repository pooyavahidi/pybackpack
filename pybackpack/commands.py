from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Union
from multiprocessing import Pool, cpu_count
from functools import partial
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import asyncio


@dataclass
class CommandResult:
    """Represents the result of a command execution.

    Attributes:
        output: The output of the command.
        succeeded: A boolean indicating whether the command was successful.
        error: The exception raised by the command if it failed.
        error_message: The error message if the command failed.
        results: List of sub-commands results if applicable.
        cancelled: A boolean indicating whether the command was cancelled. In
            case of a concurrent command, it indicates whether the task was
            cancelled or not.
        metadata: Any additional data.
    """

    output: Any = None
    succeeded: bool = True
    error: Any = None
    error_message: str = None
    results: List[CommandResult] = field(default_factory=list)
    cancelled: bool = False
    metadata: Any = None


class Command(ABC):
    """An abstract class for any executable command.It provides a simple
    interface for all classes which implement the `Command` pattern.
    """

    @abstractmethod
    def run(self, input_data: Optional[Any] = None) -> CommandResult:
        """Runs the command.

        Args:
            input_data: The input for the command. Defaults to None.

        Returns:
            CommandResult: The result of executing the command.
        """


class AsyncCommand(ABC):
    """An abstract class for any executable asynchronous command. It provides a
    simple interface for all classes which implement the `Command` pattern.
    """

    @abstractmethod
    async def async_run(
        self, input_data: Optional[Any] = None
    ) -> CommandResult:
        """Runs the command asynchronously.

        Args:
            input_data: The input for the command. Defaults to None.

        Returns:
            CommandResult: The result after executing the command.
        """


def run_command(
    command: Command, input_data: Optional[Any] = None
) -> CommandResult:
    """Runs a command synchronously and returns the result.

    Args:
        command: A Command object to be executed.
        input_data: The input for the command. Defaults to None.

    Returns:
        CommandResult: The result of executing the command.
    """
    try:
        return command.run(input_data=input_data)
    except Exception as ex:
        return CommandResult(
            output=None,
            succeeded=False,
            error=ex,
            error_message=str(ex),
        )


async def async_run_command(
    command: AsyncCommand, input_data: Optional[Any] = None
) -> CommandResult:
    """Runs a command asynchronously and returns the result.

    Args:
        command: A Command object to be executed.
        input_data: The input for the command. Defaults to None.
    Returns:
        CommandResult: The result of executing the command.
    """
    try:
        return await command.async_run(input_data=input_data)
    except Exception as ex:
        return CommandResult(
            output=None,
            succeeded=False,
            error=ex,
            error_message=str(ex),
        )


class PipeCommand(Command, AsyncCommand):
    """This is a Macro Command which runs the commands in sequence, similar to
    `Pipe` in Unix-like operating systems.
    The output of each command is provided as input to the next
    command in sequence. If any command fails, the pipeline stops and returns
    the result with success set to False. An error is raised if no commands
    list is provided.

    Attributes:
        commands: A list of Commands to be executed in the pipeline.
    """

    def __init__(
        self,
        commands: List[Union[Command, AsyncCommand]],
        collect_results=True,
    ):
        if not commands:
            raise ValueError("Commands list cannot be None or empty")
        self.commands = commands
        self._last_result = None
        self._collect_results = collect_results
        self._results = []

    def _evaluate_execution(self, result: CommandResult) -> bool:
        """Evaluates the result of a command and returns a boolean indicating
        whether the process should continue or not.

        Args:
            result: The result of the command.

        Returns:
            bool: A boolean indicating whether the pipeline has failed or not.
        """
        self._last_result = result

        if self._collect_results:
            self._results.append(result)

        return result.succeeded

    def _create_final_result(self, result: CommandResult) -> CommandResult:
        """Returns the final result of the pipeline. It returns the details of
        the last command in the pipeline.

        Args:
            result: The result of the last command in the sequence.

        Returns:
            CommandResult: The final result of the pipeline.
        """

        return CommandResult(
            output=result.output,
            succeeded=result.succeeded,
            results=self._results,
            error=result.error,
            error_message=result.error_message,
        )

    def run(self, input_data: Optional[Any] = None) -> CommandResult:
        for command in self.commands:
            if self._last_result:
                input_data = self._last_result.output

            result = run_command(command, input_data=input_data)
            if not self._evaluate_execution(result):
                break

        return self._create_final_result(result)

    async def async_run(
        self, input_data: Optional[Any] = None
    ) -> CommandResult:
        for command in self.commands:
            if self._last_result:
                input_data = self._last_result.output

            result = await async_run_command(command, input_data=input_data)
            if not self._evaluate_execution(result):
                break

        return self._create_final_result(result)


class SequentialCommand(Command, AsyncCommand):
    """This is a Macro Command which runs the commands sequentially with an
    option to set the operator between the commands.

    Each command runs after the previous command has finished in the sequence.
    The `operator` attribute sets the operation between the commands. This
    attribute is similar to the `&&`, `||` and `;` operators in Unix-like
    operating systems.

    Attributes:
        commands: A list of Command objects to be executed in sequence.
        operator: The operator between the commands which can be one of the
            following:
            - `&&`: (default), the next command will run only if the previous
            command was successful.
            - `||`: the next command will run only if the previous command
            failed.
            - None: It will act like the `;` operator, meaning the next command
            will run regardless of the outcome of the previous command.
        collect_outputs: A boolean indicating whether to collect the outputs
            of all commands. Defaults to False.
            If collect_outputs is True, it gathers outputs of all results.
            Otherwise, the output is the output of the last command.
    """

    def __init__(
        self,
        commands: List[Union[Command, AsyncCommand]],
        operator="&&",
        collect_results=True,
    ):
        if not commands:
            raise ValueError("Commands list cannot be None or empty")

        if operator not in ["&&", "||", None]:
            raise ValueError("Invalid operator")

        self.commands = commands
        self.operator = operator
        self._collect_results = collect_results
        self._results = []

    def _evaluate_execution(self, result: CommandResult) -> bool:
        """Evaluates the result of a command and returns a boolean indicating
        whether the process should continue or not.

        Args:
            result: The result of the command.

        Returns:
            bool: A boolean indicating whether to continue or not.
        """
        if self._collect_results:
            self._results.append(result)

        if not result.succeeded and self.operator == "&&":
            return False

        if result.succeeded and self.operator == "||":
            return False

        return True

    def _create_final_result(self, result: CommandResult) -> CommandResult:
        """Returns the final result of the sequence.

        Args:
            result: The result of the last command in the sequence.

        Returns:
            CommandResult: The final result of the sequence.
        """
        last_error = result.error
        last_error_message = result.error_message
        succeeded = result.succeeded
        output = [result.output for result in self._results if result]

        if not self.operator:
            # For None(;) operator, the final result is always succeeded.
            succeeded = True

        return CommandResult(
            output=output,
            succeeded=succeeded,
            results=self._results,
            error=last_error,
            error_message=last_error_message,
        )

    def run(self, input_data: Optional[Any] = None) -> CommandResult:
        for command in self.commands:
            result = run_command(command, input_data=input_data)
            if not self._evaluate_execution(result):
                break

        return self._create_final_result(result)

    async def async_run(
        self, input_data: Optional[Any] = None
    ) -> CommandResult:
        for command in self.commands:
            result = await async_run_command(command, input_data=input_data)
            if not self._evaluate_execution(result):
                break

        return self._create_final_result(result)


class MultiProcessCommand(Command):
    """This is a Macro Command which runs the commands in parallel using
    multiprocessing.

    The commands are indepndent of each other and can be run in parallel. This
    class uses the `multiprocessing.Pool` to run the commands in parallel.

    This class always collects the outputs of all commands' execution.

    Attributes:
        commands: A list of Command objects to be executed in parallel.
        pool_size: The number of concurrent processes to run the commands.
            Defaults to the number of CPUs available on the system.
    """

    def __init__(
        self,
        commands: List[Command],
        pool_size: int = None,
    ):
        if not commands:
            raise ValueError("Commands list cannot be None or empty")

        self.commands = commands
        self._pool_size = pool_size or cpu_count()

    def run(self, input_data: Optional[Any] = None) -> CommandResult:
        # Create a new function with input_data pre-filled
        run_command_with_input = partial(run_command, input_data=input_data)

        with Pool(self._pool_size) as pool:
            results = pool.map(run_command_with_input, self.commands)

        outputs = [result.output for result in results if result]

        return CommandResult(output=outputs, results=results)


class AsyncAdapterCommand(AsyncCommand):
    """This is an Adapter command which converts the interface of a command
    from synchronous to asynchronous.

    Attributes:
        command: A synchronous Command object to be converted to an
            asynchronous command.
    """

    def __init__(self, command: AsyncCommand):
        self.command = command

    async def async_run(
        self, input_data: Optional[Any] = None
    ) -> CommandResult:
        return await asyncio.to_thread(run_command, self.command, input_data)


class SyncAdapterCommand(Command):
    """This is an Adapter command which converts the interface of a command
    from asynchronous to synchronous.

    Attributes:
        command: An asynchronous Command object to be converted to an
            synchronous command.

    """

    def __init__(self, command: Command):
        self.command = command

    def run(self, input_data: Optional[Any] = None) -> CommandResult:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    async_run_command(
                        command=self.command, input_data=input_data
                    )
                )
            )
            return future.result()


class AsyncConcurrentCommand(AsyncCommand):
    """This is a Macro Command which runs the asynchronous commands
    concurrently and wait for all commands to finish before returning the
    result.

    The commands are indepndent of each other and can be run in parallel. This
    class uses the `asyncio` to run the commands in parallel and wait for all
    completion (or cancellation of all running tasks).

    This class always collects the outputs of all commands' execution. the
    `results` attribute contains the results of all commands regardless of
    their success or failure. In case of cancellation, the result for the
    cancelled tasks will be set to None.

    Attributes:
        commands: A list of Command objects to be executed in asynchronously.
        concurrency_limit: The number of concurrent executions.
            This will help limit the number of concurrent tasks to avoid
            overloading the underlying system.
            Defaults to 0, which means no limit.
        callback: A callback function to be called after each command execution.
            The callback function should accept three arguments:
                - command: The command that was executed.
                - input_data: The input data that was passed to the command.
                - result: The result of the command execution.
            The callback function should not return anything.
        cancel_on_failure: A boolean indicating whether to cancel all running
            tasks by the first encountered failure. Defaults to False.

    """

    def __init__(
        self,
        commands: List[AsyncCommand],
        concurrency_limit: int = 0,
        callback=None,
        cancel_on_failure: bool = False,
    ):
        self.commands = commands
        self._concurrency_limit = concurrency_limit
        self._callback = callback
        self._cancel_on_failure = cancel_on_failure

    async def _async_run_task(
        self,
        command: Command,
        input_data: Any,
        semaphore: asyncio.Semaphore,
        callback,
    ) -> CommandResult:
        try:
            if semaphore:
                async with semaphore:
                    result = await async_run_command(
                        command=command, input_data=input_data
                    )
            else:
                result = await async_run_command(
                    command=command, input_data=input_data
                )
        except Exception as ex:
            result = CommandResult(
                output=None,
                succeeded=False,
                error=ex,
                error_message=str(ex),
            )

        if callback:
            callback(command=command, input_data=input_data, result=result)

        return result, command

    async def async_run(
        self, input_data: Optional[Any] = None
    ) -> CommandResult:
        if self._concurrency_limit > 0:
            semaphore = asyncio.Semaphore(self._concurrency_limit)
        else:
            semaphore = None

        tasks = [
            asyncio.create_task(
                self._async_run_task(
                    command=command,
                    input_data=input_data,
                    semaphore=semaphore,
                    callback=self._callback,
                )
            )
            for command in self.commands
        ]

        # Initialize a list to store the results in order
        results = [None] * len(self.commands)
        error = None

        try:
            for task in asyncio.as_completed(tasks):
                result, command = await task

                # Find the index of the command and store the result at that
                # index in the results list
                index = self.commands.index(command)
                results[index] = result

                if not result.succeeded and self._cancel_on_failure:
                    raise RuntimeError(
                        (
                            f"Task failed with error {result.error_message}. "
                            "Cancelling all running tasks."
                        )
                    )
        except Exception as ex:
            error = ex
            # Cancel all running tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Wait for all tasks to be cancelled or completed
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # Create the final result. If the task was cancelled, the result will be
        # set to a failed result with cancelled=True
        final_results = []
        for res in results:
            if not res:
                final_result = CommandResult(succeeded=False, cancelled=True)
            else:
                final_result = res
            final_results.append(final_result)

        outputs = [res.output for res in final_results]

        return CommandResult(
            output=outputs,
            results=final_results,
            succeeded=not error,
            error=error,
            error_message=str(error) if error else None,
        )
