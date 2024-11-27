from functools import wraps
from typeguard import typechecked


def format_checked(func):
    """
    Decorator for checking types and non-emptiness of specific argument types for a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with type and non-emptiness checks.
    """
    checked_func = typechecked(func)
    function_name = func.__name__
    types_to_check = (str, list, dict, tuple, set)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # After type checking, check for non-emptiness
        for idx, arg in enumerate(args):
            arg_name = func.__code__.co_varnames[idx]
            if isinstance(arg, types_to_check) and len(arg) == 0:
                raise ValueError(f"Argument '{arg_name}' in function '{function_name}' has zero length")
        for key, value in kwargs.items():
            if isinstance(value, types_to_check) and len(value) == 0:
                raise ValueError(f"Argument '{key}' in function '{function_name}' is empty")
        return checked_func(*args, **kwargs)

    return wrapper


# 添加 debug_print 函数（假设这是一个自定义的日志函数）
def debug_print(message):
    print(f'DEBUG: {message}')


@format_checked
def parse_query_into_attributes(query: str):
    """
    解析查询，提取关键属性。

    Args:
        query (str): 查询字符串。

    Returns:
        dict: 包含解析后的属性的字典。
    """
    attributes = ["subject", "action", "object", "location"]
    parsed_attributes = parse_query(query, attributes)
    debug_print(f'Parsed Query: {parsed_attributes}')
    return parsed_attributes


@format_checked
def filter_non_empty_strings(strings: list):
    """
    过滤掉列表中的空字符串。

    Args:
        strings (list): 字符串列表。

    Returns:
        list: 过滤后的字符串列表。
    """
    return [s for s in strings if len(s) > 0]


# 示例 parse_query 函数（假设这是你需要的解析函数）
def parse_query(query: str, attributes: list) -> dict:
    """
    示例解析函数，实际应用中应替换为真实的解析逻辑。

    Args:
        query (str): 查询字符串。
        attributes (list): 需要解析的属性列表。

    Returns:
        dict: 包含解析后的属性的字典。
    """
    # 示例解析逻辑
    parsed_dict = {}
    for attr in attributes:
        parsed_dict[attr] = f"parsed_{attr}"
    return parsed_dict