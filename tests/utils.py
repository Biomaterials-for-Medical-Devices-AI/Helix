from enum import StrEnum
from streamlit.testing.v1 import AppTest


def get_element_by_key(at: AppTest, element_type: str, key: StrEnum):
    if elements := at.get(element_type):
        for element in elements:
            if element.key == key:
                return element
    else:
        return None


def get_element_by_label(at: AppTest, element_type: str, label: str):
    if elements := at.get(element_type):
        for element in elements:
            if element.label == label:
                return element
    else:
        return None
