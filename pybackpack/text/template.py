from typing import Dict, Set
from jinja2 import Environment, StrictUndefined, meta
from jinja2.exceptions import UndefinedError


# Configure the jinja2 environments.
# Since in most cases a small groups of environments can be reused, we will
# configure them here and reuse them.
strict_env = Environment(undefined=StrictUndefined)


class Template:
    """A template class to render text using jinja2.

    Attributes:
        text (str): The text to be rendered.
        env (Environment): The jinja2 environment to be used. If note provided,
            the default environment with strict undefined variables will be
            used.
    """

    def __init__(self, text: str, env: Environment = None):
        if text is None:
            raise ValueError("text cannot be None")
        self.text = text

        self.env = env
        if not self.env:
            self.env = strict_env

    def render(self, variables: Dict = None):
        """Renders the text using jinja2.

        Args:
            variables: The variables to be used in the text. Defaults to None.
        """

        jinja_template = self.env.from_string(self.text)

        try:
            if variables:
                return jinja_template.render(**variables)
            return jinja_template.render()

        except UndefinedError as ex:
            raise ValueError("Undefined variable in template.") from ex
        except Exception as ex:
            raise ex

    def find_variables(self) -> Set[str]:
        """Find all the referenced variables from the text.

        Returns:
            Set[str]: The set of referenced variables.
        """
        ast = self.env.parse(self.text)
        return meta.find_undeclared_variables(ast)
