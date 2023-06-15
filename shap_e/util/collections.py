from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

class AttrDict(OrderedDict):
    """
    An attribute dictionary that automatically handles nested keys joined by "/".

    Originally copied from: https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """

    MARKER = object()

    def __init__(self, *args, **kwargs):
        """
        Initializes the AttrDict object.

        Args:
            *args: Variable length arguments. If a single argument is provided and it's either a dict or an AttrDict,
                the items of the dictionary are used to populate the AttrDict.
            **kwargs: Keyword arguments. The key-value pairs are used to populate the AttrDict.
        """
        if len(args) == 0:
            for key, value in kwargs.items():
                self.__setitem__(key, value)
        else:
            assert len(args) == 1
            assert isinstance(args[0], (dict, AttrDict))
            for key, value in args[0].items():
                self.__setitem__(key, value)

    def __contains__(self, key):
        """
        Checks if the key is contained in the AttrDict.

        Args:
            key (Any): The key to check.

        Returns:
            bool: True if the key is present in the AttrDict, False otherwise.
        """
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            return key in self and next_key in self[key]
        return super(AttrDict, self).__contains__(key)

    def __setitem__(self, key, value):
        """
        Sets the value for the given key in the AttrDict.

        Args:
            key (Any): The key to set.
            value (Any): The value to assign to the key.
        """
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            if key not in self:
                self[key] = AttrDict()
            self[key].__setitem__(next_key, value)
            return

        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(**value)
        if isinstance(value, list):
            value = [AttrDict(val) if isinstance(val, dict) else val for val in value]
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        """
        Returns the value for the given key from the AttrDict.

        Args:
            key (Any): The key to retrieve the value for.

        Returns:
            Any: The value associated with the key.
        """
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            val = self[key]
            if not isinstance(val, AttrDict):
                raise ValueError
            return val.__getitem__(next_key)

        return self.get(key, None)

    def all_keys(
        self,
        leaves_only: bool = False,
        parent: Optional[str] = None,
    ) -> List[str]:
        """
        Returns a list of all keys in the AttrDict.

        Args:
            leaves_only (bool): If True, only includes the keys that do not have nested dictionaries as values.
            parent (Optional[str]): The parent key to prepend to each key in the result.

        Returns:
            List[str]: A list of all keys in the AttrDict.
        """
        keys = []
        for key in self.keys():
            cur = key if parent is None else f"{parent}/{key}"
            if not leaves_only or not isinstance(self[key], dict):
                keys.append(cur)
            if isinstance(self[key], dict):
                keys.extend(self[key].all_keys(leaves_only=leaves_only, parent=cur))
        return keys

    def dumpable(self, strip=True):
        """
        Casts the AttrDict into an OrderedDict and removes internal attributes.

        Args:
            strip (bool): If True, removes attributes starting with an underscore.

        Returns:
            OrderedDict: An ordered dictionary representation of the AttrDict.
        """
        def _dump(val):
            if isinstance(val, AttrDict):
                return val.dumpable()
            elif isinstance(val, list):
                return [_dump(v) for v in val]
            return val

        if strip:
            return {k: _dump(v) for k, v in self.items() if not k.startswith("_")}
        return {k: _dump(v if not k.startswith("_") else repr(v)) for k, v in self.items()}

    def map(
        self,
        map_fn: Callable[[Any, Any], Any],
        should_map: Optional[Callable[[Any, Any], bool]] = None,
    ) -> "AttrDict":
        """
        Creates a copy of the AttrDict where some or all values are transformed by the map_fn.

        Args:
            map_fn (Callable[[Any, Any], Any]): The function to apply to the values.
            should_map (Optional[Callable[[Any, Any], bool]]): If provided, only applies the map_fn to those values that evaluate to True.

        Returns:
            AttrDict: A copy of the AttrDict with the transformed values.
        """
        def _apply(key, val):
            if isinstance(val, AttrDict):
                return val.map(map_fn, should_map)
            elif should_map is None or should_map(key, val):
                return map_fn(key, val)
            return val

        return AttrDict({k: _apply(k, v) for k, v in self.items()})

    def __eq__(self, other):
        """
        Checks the equality of two AttrDict objects.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return self.keys() == other.keys() and all(self[k] == other[k] for k in self.keys())

    def combine(
        self,
        other: Dict[str, Any],
        combine_fn: Callable[[Optional[Any], Optional[Any]], Any],
    ) -> "AttrDict":
        """
        Combines the values of two dictionaries with the same structure using a combine_fn.

        Args:
            other (Dict[str, Any]): The dictionary to combine with.
            combine_fn (Callable[[Optional[Any], Optional[Any]], Any]): A function to combine the values.

        Returns:
            AttrDict: A new AttrDict with the combined values.
        """
        def _apply(val, other_val):
            if val is not None and isinstance(val, AttrDict):
                assert isinstance(other_val, AttrDict)
                return val.combine(other_val, combine_fn)
            return combine_fn(val, other_val)

        # TODO nit: this changes the ordering..
        keys = self.keys() | other.keys()
        return AttrDict({k: _apply(self[k], other[k]) for k in keys})

    # Assigns the methods __setitem__ and __getitem__ to the special methods __setattr__ and __getattr__ respectively.

    # In Python, __setattr__ and __getattr__ are special methods that are called when an attribute is set or accessed, respectively.
    # By default, these methods are used for setting and getting regular attributes of an object.

    # The assignments link the behaviors of setting and getting attributes with the behaviors of setting and getting items in the dictionary-like structure of the AttrDict class.
    # It means that when an attribute is set or accessed using dot notation (obj.attr), it will actually call the corresponding __setitem__ or __getitem__ methods, allowing the object to behave like a dictionary.

    # This assignment enables accessing and modifying the attributes of an AttrDict instance using both dot notation and dictionary-like item access ([]).

    # For example:
    # obj = AttrDict()
    # obj.key = "value"  # Calls __setitem__ method, equivalent to obj["key"] = "value"
    # print(obj.key)  # Calls __getitem__ method, equivalent to print(obj["key"])

    # By making these assignments, the AttrDict class extends the functionality of regular attribute setting and getting to support dictionary-like access and assignment, providing a more flexible interface for accessing and manipulating the data stored in the AttrDict object.

    __setattr__, __getattr__ = __setitem__, __getitem__
