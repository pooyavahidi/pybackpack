import os
import subprocess
from typing import List, Any
from abc import ABC, abstractmethod
from pybackpack.commands import Command, CommandResult, run_command


# pylint: disable=too-many-arguments,too-many-instance-attributes
class ProcessCommand(Command):
    """This class is a wrapper around subprocess.run() to run shell commands.

    The result of the command is stored in the `result` attribute. Also the
    stdout of the command is returned by the `run()` method.
    Attributes:
        cmd (list): The command to run.
        capture_output (bool): Capture stdout and stderr.
        check (bool): If True, raise an exception if the command fails.
        text (bool): If True, stdout and stderr are returned as strings.
        timeout (int): The timeout for the command.
        cwd (str): The current working directory.
        shell (bool): If True, run the command in a shell.
        encoding (str): The encoding of the command output.
        errors (str): The error handling of the command output.
        stdin (int): The stdin of the command.
        env (dict): The environment variables for the command.
        inherit_env (bool): If True, use the current environment as the base.
        other_popen_kwargs (dict): Keyword arguments for subprocess.Popen.
    """

    def __init__(
        self,
        cmd,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        timeout=None,
        cwd=None,
        shell=False,
        errors=None,
        stdin=None,
        env=None,
        inherit_env=True,
        **other_popen_kwargs,
    ):
        if inherit_env:
            # Use the current environment as the base
            self.env = os.environ.copy()
            if env:
                # Update with provided env variables
                self.env.update(env)
        else:
            self.env = env

        self.cmd = cmd
        self.capture_output = capture_output
        self.check = check
        self.text = text
        self.timeout = timeout
        self.cwd = cwd
        self.shell = shell
        self.encoding = encoding
        self.errors = errors
        self.stdin = stdin
        self.other_popen_kwargs = other_popen_kwargs

    def run(self, input_data=None) -> CommandResult:
        metadata = {}
        try:
            result = subprocess.run(
                self.cmd,
                capture_output=self.capture_output,
                check=self.check,
                text=self.text,
                timeout=self.timeout,
                cwd=self.cwd,
                env=self.env,
                shell=self.shell,
                encoding=self.encoding,
                errors=self.errors,
                stdin=self.stdin,
                input=input_data,
                **self.other_popen_kwargs,
            )
            cmd_result = CommandResult(
                output=result.stdout,
                metadata=result,
            )

        except Exception as ex:
            if isinstance(ex, subprocess.TimeoutExpired):
                metadata["failure_reason"] = "timeout"
            if isinstance(ex, FileNotFoundError):
                metadata["failure_reason"] = "command_not_found"

            cmd_result = CommandResult(
                output=None,
                succeeded=False,
                error=ex,
                error_message=str(ex),
                metadata=metadata,
            )

        return cmd_result


class ProcessOutputParser(ABC):
    """Parse the command output and return the result in python object."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse the command output and return the result in python object."""


class DefaultOutputParser(ProcessOutputParser):
    """Default output parser. Return the output as a list of lines."""

    def parse(self, output: str) -> Any:
        if not output:
            return []

        # Use strip() twice to remove empty lines
        return [line.strip() for line in output.splitlines() if line.strip()]


def run_shell_command(
    cmd: List[str],
    output_parser: ProcessOutputParser = DefaultOutputParser(),
    **kwargs,
) -> Any:
    """Run a command and return the output as a list of lines. This is a
    simple wrapper around ProcessCommand. For more control over the command
    execution, or use it with pipes, sequenes, etc. use ProcessCommand
    directly.

    Note: This function is using ProcessCommand with default values. By default
    shell is False, so the command is not executed in a shell.

    Args:
        cmd: The command to run. Command must be a list of strings.
        output_parser: The output parser to use. Default is DefaultOutputParser.
        kwargs: Keyword arguments for ProcessCommand.
    Returns:
        The output based on the given output_parser.
    """
    process_cmd = ProcessCommand(cmd, **kwargs)
    result = run_command(process_cmd)

    if not result.succeeded:
        raise result.error

    return output_parser.parse(result.output)
