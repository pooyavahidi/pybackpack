import pytest
from pybackpack.text.template import Template

# pylint: disable=invalid-name


def test_render_text():
    # test plain or empty text, returns as is
    assert Template("foo").render() == "foo"
    assert Template("").render() == ""
    assert Template(" ").render() == " "

    # None as text raisee error
    with pytest.raises(ValueError):
        Template(None)

    # Pass variables
    template = Template("foo {{ var1 }}")
    variables = {"var1": "baz"}
    assert template.render(variables) == "foo baz"

    # Pass variables with a text which has no variables
    template = Template("foo")
    variables = {"var1": "baz"}
    assert template.render(variables) == "foo"

    # Passing None variables with a text which has variables raises ValueError
    template = Template("foo {{ var1 }}")
    with pytest.raises(ValueError):
        template.render(variables=None)


def test_render_text_conditional_statements():
    # Pass no variables with a text which has a default for a variable
    template = Template("foo {{ var1 | default('baz') }}")
    assert template.render(variables=None) == "foo baz"
    assert template.render(variables={}) == "foo baz"
    # Now passing a variable will override the default
    assert template.render(variables={"var1": "qux"}) == "foo qux"

    # undefined variables also can be checked using if-else statements
    template = Template(
        "foo{% if var1 is defined %} {{var1}}{% else %} qux{% endif %}"
    )
    assert template.render(variables=None) == "foo qux"
    assert template.render(variables={}) == "foo qux"
    assert template.render(variables={"var1": "baz"}) == "foo baz"

    # if-else: test jinja tempalte with conditional statement
    # test jinja tempalte with conditional statement
    template = Template("foo {% if var1 %}baz{% endif %}")
    variables = {"var1": True}
    assert template.render(variables) == "foo baz"

    # if-else: Another way to test the same conditional statement
    template = Template("foo {{ 'baz' if var1 else 'qux' }}")
    variables = {"var1": False}
    assert template.render(variables) == "foo qux"


def test_template_variables():
    # Test finding variables from a template
    template = Template("foo {{ var1 }}")
    assert template.find_variables() == {"var1"}

    # Test finding variables from a template with a default value
    template = Template("foo {{ var1 | default('baz') }}")
    assert template.find_variables() == {"var1"}
