from enum import StrEnum

from streamlit.testing.v1 import AppTest


def get_element_by_key(at: AppTest, element_type: str, key: StrEnum):
    """Get a page element for testing based on its key.

    Returns `None` if no match is found.

    Args:
        at (AppTest): The AppTest instance.
        element_type (str): The name of the type of element.
        key (StrEnum): The key of the element you want.

    Returns:
        Node | None: The matching element of
    """
    if elements := at.get(element_type):
        for element in elements:
            if element.key == key:
                return element
    else:
        return None


def get_element_by_label(at: AppTest, element_type: str, label: str):
    """Get a page element for testing based on its label.

    Returns `None` if no match is found.

    Args:
        at (AppTest): The AppTest instance.
        element_type (str): The name of the type of element.
        key (str): The label of the element you want.

    Returns:
        Node | None: The matching element of
    """
    if elements := at.get(element_type):
        for element in elements:
            if element.label == label:
                return element
    else:
        return None
