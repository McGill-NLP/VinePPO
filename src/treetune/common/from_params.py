import collections.abc
import inspect
from copy import deepcopy
from pathlib import Path
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Optional,
)

from treetune import logging_utils
from treetune.common.lazy import Lazy
from treetune.common.params import Params

logger = logging_utils.get_logger(__name__)

T = TypeVar("T", bound="FromParams")

# If a function parameter has no default value specified,
# this is what the inspect module returns.
_NO_DEFAULT = inspect.Parameter.empty

import collections
import hashlib
import io
from typing import Any, MutableMapping

import base58
import dill


class CustomDetHash:
    def det_hash_object(self) -> Any:
        """
        By default, `det_hash()` pickles an object, and returns the hash of the pickled
        representation. Sometimes you want to take control over what goes into
        that hash. In that case, implement this method. `det_hash()` will pickle the
        result of this method instead of the object itself.

        If you return `None`, `det_hash()` falls back to the original behavior and pickles
        the object.
        """
        raise NotImplementedError()


class DetHashFromInitParams(CustomDetHash):
    """
    Add this class as a mixin base class to make sure your class's det_hash is derived
    exclusively from the parameters passed to __init__().
    """

    _det_hash_object: Any

    def __new__(cls, *args, **kwargs):
        super_new = super(DetHashFromInitParams, cls).__new__
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            instance = super_new(cls)
        else:
            instance = super_new(cls, *args, **kwargs)
        instance._det_hash_object = (args, kwargs)
        return instance

    def det_hash_object(self) -> Any:
        return self._det_hash_object


class DetHashWithVersion(CustomDetHash):
    """
    Add this class as a mixing base class to make sure your class's det_hash can be modified
    by altering a static `VERSION` member of your class.
    """

    VERSION = None

    def det_hash_object(self) -> Any:
        if self.VERSION is not None:
            return self.VERSION, self
        else:
            return None


class _DetHashPickler(dill.Pickler):
    def __init__(self, buffer: io.BytesIO):
        super().__init__(buffer)

        # We keep track of how deeply we are nesting the pickling of an object.
        # If a class returns `self` as part of `det_hash_object()`, it causes an
        # infinite recursion, because we try to pickle the `det_hash_object()`, which
        # contains `self`, which returns a `det_hash_object()`, etc.
        # So we keep track of how many times recursively we are trying to pickle the
        # same object. We only call `det_hash_object()` the first time. We assume that
        # if `det_hash_object()` returns `self` in any way, we want the second time
        # to just pickle the object as normal. `DetHashWithVersion` takes advantage
        # of this ability.
        self.recursively_pickled_ids: MutableMapping[int, int] = collections.Counter()

    def save(self, obj, save_persistent_id=True):
        self.recursively_pickled_ids[id(obj)] += 1
        super().save(obj, save_persistent_id)
        self.recursively_pickled_ids[id(obj)] -= 1

    def persistent_id(self, obj: Any) -> Any:
        if (
            isinstance(obj, CustomDetHash)
            and self.recursively_pickled_ids[id(obj)] <= 1
        ):
            det_hash_object = obj.det_hash_object()
            if det_hash_object is not None:
                return (
                    obj.__class__.__module__,
                    obj.__class__.__qualname__,
                    det_hash_object,
                )
            else:
                return None
        elif isinstance(obj, type):
            return obj.__module__, obj.__qualname__
        else:
            return None


def det_hash(o: Any) -> str:
    """
    Returns a deterministic hash code of arbitrary Python objects.

    If you want to override how we calculate the deterministic hash, derive from the
    `CustomDetHash` class and implement `det_hash_object()`.
    """
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        pickler = _DetHashPickler(buffer)
        pickler.dump(o)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


def takes_arg(obj, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return arg in signature.parameters


def takes_kwargs(obj) -> bool:
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD  # type: ignore
        for p in signature.parameters.values()
    )


def can_construct_from_params(type_: Type) -> bool:
    if type_ in [str, int, float, bool]:
        return True
    origin = getattr(type_, "__origin__", None)
    if origin == Lazy:
        return True
    elif origin:
        if hasattr(type_, "from_params"):
            return True
        args = getattr(type_, "__args__")
        return all(can_construct_from_params(arg) for arg in args)

    return hasattr(type_, "from_params")


def is_base_registrable(cls) -> bool:
    """
    Checks whether this is a class that directly inherits from Registrable, or is a subclass of such
    a class.
    """
    from treetune.common.registrable import Registrable  # import here to avoid circular imports

    if not issubclass(cls, Registrable):
        return False
    method_resolution_order = inspect.getmro(cls)[1:]
    for base_class in method_resolution_order:
        if issubclass(base_class, Registrable) and base_class is not Registrable:
            return False
    return True


def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin == Union:
        return Union[tuple([arg for arg in args if arg != type(None)])]  # noqa: E721
    else:
        return annotation


def infer_constructor_params(
    cls: Type[T], constructor: Union[Callable[..., T], Callable[[T], None]] = None
) -> Dict[str, inspect.Parameter]:
    if constructor is None:
        constructor = cls.__init__
    return infer_method_params(cls, constructor)


infer_params = infer_constructor_params  # Legacy name


def infer_method_params(cls: Type[T], method: Callable) -> Dict[str, inspect.Parameter]:
    signature = inspect.signature(method)
    parameters = dict(signature.parameters)

    has_kwargs = False
    var_positional_key = None
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == param.VAR_POSITIONAL:
            var_positional_key = param.name

    if var_positional_key:
        del parameters[var_positional_key]

    if not has_kwargs:
        return parameters

    # "mro" is "method resolution order". The first one is the current class, the next is the
    # first superclass, and so on. We take the first superclass we find that inherits from
    # FromParams.
    super_class = None
    for super_class_candidate in cls.mro()[1:]:
        if issubclass(super_class_candidate, FromParams):
            super_class = super_class_candidate
            break
    if super_class:
        super_parameters = infer_params(super_class)
    else:
        super_parameters = {}

    return {
        **super_parameters,
        **parameters,
    }  # Subclass parameters overwrite superclass ones


def create_kwargs(
    constructor: Callable[..., T], cls: Type[T], params: Params, **extras
) -> Dict[str, Any]:
    """
    Given some class, a `Params` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the `extras` will just be used as is.
    For instance, you might provide an existing `Vocabulary` this way.
    """
    # Get the signature of the constructor.

    kwargs: Dict[str, Any] = {}

    parameters = infer_params(cls, constructor)
    accepts_kwargs = False

    # Iterate over all the constructor parameters and their annotations.
    for param_name, param in parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if param_name == "self":
            continue

        if param.kind == param.VAR_KEYWORD:
            # When a class takes **kwargs, we do two things: first, we assume that the **kwargs are
            # getting passed to the super class, so we inspect super class constructors to get
            # allowed arguments (that happens in `infer_params` above).  Second, we store the fact
            # that the method allows extra keys; if we get extra parameters, instead of crashing,
            # we'll just pass them as-is to the constructor, and hope that you know what you're
            # doing.
            accepts_kwargs = True
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)

        explicitly_set = param_name in params
        constructed_arg = pop_and_construct_arg(
            cls.__name__, param_name, annotation, param.default, params, **extras
        )

        # If the param wasn't explicitly set in `params` and we just ended up constructing
        # the default value for the parameter, we can just omit it.
        # Leaving it in can cause issues with **kwargs in some corner cases, where you might end up
        # with multiple values for a single parameter (e.g., the default value gives you lazy=False
        # for a dataset reader inside **kwargs, but a particular dataset reader actually hard-codes
        # lazy=True - the superclass sees both lazy=True and lazy=False in its constructor).
        if explicitly_set or constructed_arg is not param.default:
            kwargs[param_name] = constructed_arg

    if accepts_kwargs:
        kwargs.update(params)
    else:
        params.assert_empty(cls.__name__)
    return kwargs


def create_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dictionary of extra arguments, returns a dictionary of
    kwargs that actually are a part of the signature of the cls.from_params
    (or cls) method.
    """
    subextras: Dict[str, Any] = {}
    if hasattr(cls, "from_params"):
        from_params_method = cls.from_params  # type: ignore
    else:
        # In some rare cases, we get a registered subclass that does _not_ have a
        # from_params method (this happens with Activations, for instance, where we
        # register pytorch modules directly).  This is a bit of a hack to make those work,
        # instead of adding a `from_params` method for them somehow. Then the extras
        # in the class constructor are what we are looking for, to pass on.
        from_params_method = cls
    if takes_kwargs(from_params_method):
        # If annotation.params accepts **kwargs, we need to pass them all along.
        # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
        # object, but `TextFieldEmbedder.from_params` does not.
        subextras = extras
    else:
        # Otherwise, only supply the ones that are actual args; any additional ones
        # will cause a TypeError.
        subextras = {
            k: v for k, v in extras.items() if takes_arg(from_params_method, k)
        }
    return subextras


def pop_and_construct_arg(
    class_name: str,
    argument_name: str,
    annotation: Type,
    default: Any,
    params: Params,
    **extras,
) -> Any:
    """
    Does the work of actually constructing an individual argument for
    [`create_kwargs`](./#create_kwargs).

    Here we're in the inner loop of iterating over the parameters to a particular constructor,
    trying to construct just one of them.  The information we get for that parameter is its name,
    its type annotation, and its default value; we also get the full set of `Params` for
    constructing the object (which we may mutate), and any `extras` that the constructor might
    need.

    We take the type annotation and default value here separately, instead of using an
    `inspect.Parameter` object directly, so that we can handle `Union` types using recursion on
    this method, trying the different annotation types in the union in turn.
    """
    # from allennlp.models.archival import load_archive  # import here to avoid circular imports

    # We used `argument_name` as the method argument to avoid conflicts with 'name' being a key in
    # `extras`, which isn't _that_ unlikely.  Now that we are inside the method, we can switch back
    # to using `name`.
    name = argument_name

    # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
    # We check the provided `extras` for these and just use them if they exist.
    if name in extras:
        if name not in params:
            return extras[name]
        else:
            logger.warning(
                f"Parameter {name} for class {class_name} was found in both "
                "**extras and in params. Using the specification found in params, "
                "but you probably put a key in a config file that you didn't need, "
                "and if it is different from what we get from **extras, you might "
                "get unexpected behavior."
            )
    # Next case is when argument should be loaded from pretrained archive.
    elif (
        name in params
        and isinstance(params.get(name), Params)
        and "_pretrained" in params.get(name)
    ):
        # load_module_params = params.pop(name).pop("_pretrained")
        # archive_file = load_module_params.pop("archive_file")
        # module_path = load_module_params.pop("module_path")
        # freeze = load_module_params.pop("freeze", True)
        # archive = load_archive(archive_file)
        # result = archive.extract_module(module_path, freeze)
        # if not isinstance(result, annotation):
        #     raise ConfigurationError(
        #         f"The module from model at {archive_file} at path {module_path} "
        #         f"was expected of type {annotation} but is of type {type(result)}"
        #     )
        raise NotImplementedError()
        return None

    popped_params = (
        params.pop(name, default) if default != _NO_DEFAULT else params.pop(name)
    )
    if popped_params is None:
        return None

    return construct_arg(class_name, name, popped_params, annotation, default, **extras)


class Step:
    pass


class _RefStep:
    pass


def construct_arg(
    class_name: str,
    argument_name: str,
    popped_params: Params,
    annotation: Type,
    default: Any,
    could_be_step: bool = True,
    **extras,
) -> Any:
    """
    The first two parameters here are only used for logging if we encounter an error.
    """

    if could_be_step:
        # We try parsing as a step _first_. Parsing as a non-step always succeeds, because
        # it will fall back to returning a dict. So we can't try parsing as a non-step first.
        backup_params = deepcopy(popped_params)
        try:
            return construct_arg(
                class_name,
                argument_name,
                popped_params,
                Step[annotation],  # type: ignore
                default,
                could_be_step=False,
                **extras,
            )
        except (ValueError, TypeError, ConfigurationError, AttributeError):
            popped_params = backup_params

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", [])

    # The parameter is optional if its default value is not the "no default" sentinel.
    optional = default != _NO_DEFAULT

    if hasattr(annotation, "from_params"):
        if popped_params is default:
            return default
        elif popped_params is not None:
            # Our params have an entry for this, so we use that.

            subextras = create_extras(annotation, extras)

            # In some cases we allow a string instead of a param dict, so
            # we need to handle that case separately.
            if isinstance(popped_params, str):
                if origin != Step:
                    # We don't allow single strings to be upgraded to steps.
                    # Since we try everything as a step first, upgrading strings to
                    # steps automatically would cause confusion every time a step
                    # name conflicts with any string anywhere in a config.
                    popped_params = Params({"type": popped_params})
            elif isinstance(popped_params, dict):
                popped_params = Params(popped_params)
            result = annotation.from_params(params=popped_params, **subextras)

            if isinstance(result, Step):
                if isinstance(result, _RefStep):
                    existing_steps: Dict[str, Step] = extras.get("existing_steps", {})
                    try:
                        result = existing_steps[result.ref()]
                    except KeyError:
                        raise _RefStep.MissingStepError(result.ref())

                expected_return_type = args[0]
                return_type = inspect.signature(result.run).return_annotation
                if return_type == inspect.Signature.empty:
                    logger.warning(
                        "Step %s has no return type annotation. Those are really helpful when "
                        "debugging, so we recommend them highly.",
                        result.__class__.__name__,
                    )
                elif not issubclass(return_type, expected_return_type):
                    raise ConfigurationError(
                        f"Step {result.name} returns {return_type}, but "
                        f"we expected {expected_return_type}."
                    )

            return result
        elif not optional:
            # Not optional and not supplied, that's an error!
            raise ConfigurationError(f"expected key {argument_name} for {class_name}")
        else:
            return default

    # If the parameter type is a Python primitive, just pop it off
    # using the correct casting pop_xyz operation.
    elif annotation in {int, bool}:
        if type(popped_params) in {int, bool}:
            return annotation(popped_params)
        else:
            raise TypeError(f"Expected {argument_name} to be a {annotation.__name__}.")
    elif annotation == str:
        # Strings are special because we allow casting from Path to str.
        if type(popped_params) == str or isinstance(popped_params, Path):
            return str(popped_params)  # type: ignore
        else:
            raise TypeError(f"Expected {argument_name} to be a string.")
    elif annotation == float:
        # Floats are special because in Python, you can put an int wherever you can put a float.
        # https://mypy.readthedocs.io/en/stable/duck_type_compatibility.html
        if type(popped_params) in {int, float}:
            return popped_params
        else:
            raise TypeError(f"Expected {argument_name} to be numeric.")

    # This is special logic for handling types like Dict[str, TokenIndexer],
    # List[TokenIndexer], Tuple[TokenIndexer, Tokenizer], and Set[TokenIndexer],
    # which it creates by instantiating each value from_params and returning the resulting structure.
    elif (
        origin in {collections.abc.Mapping, Mapping, Dict, dict}
        and len(args) == 2
        and can_construct_from_params(args[-1])
    ):
        value_cls = annotation.__args__[-1]
        value_dict = {}
        if not isinstance(popped_params, Mapping):
            raise TypeError(
                f"Expected {argument_name} to be a Mapping (probably a dict or a Params object)."
            )

        for key, value_params in popped_params.items():
            value_dict[key] = construct_arg(
                str(value_cls),
                argument_name + "." + key,
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )

        return value_dict

    elif origin in (Tuple, tuple) and all(
        can_construct_from_params(arg) for arg in args
    ):
        value_list = []

        for i, (value_cls, value_params) in enumerate(
            zip(annotation.__args__, popped_params)
        ):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return tuple(value_list)

    elif origin in (Set, set) and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]

        value_set = set()

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_set.add(value)

        return value_set

    elif origin == Union:
        # Storing this so we can recover it later if we need to.
        backup_params = deepcopy(popped_params)

        # We'll try each of the given types in the union sequentially, returning the first one that
        # succeeds.
        error_chain: Optional[Exception] = None
        for arg_annotation in args:
            try:
                return construct_arg(
                    str(arg_annotation),
                    argument_name,
                    popped_params,
                    arg_annotation,
                    default,
                    **extras,
                )
            except (ValueError, TypeError, ConfigurationError, AttributeError) as e:
                # Our attempt to construct the argument may have modified popped_params, so we
                # restore it here.
                popped_params = deepcopy(backup_params)
                e.args = (
                    f"While constructing an argument of type {arg_annotation}",
                ) + e.args
                e.__cause__ = error_chain
                error_chain = e

        # If none of them succeeded, we crash.
        config_error = ConfigurationError(
            f"Failed to construct argument {argument_name} with type {annotation}."
        )
        config_error.__cause__ = error_chain
        raise config_error
    elif origin == Lazy:
        if popped_params is default:
            return default

        value_cls = args[0]
        subextras = create_extras(value_cls, extras)
        return Lazy(value_cls, params=deepcopy(popped_params), constructor_extras=subextras)  # type: ignore

    # For any other kind of iterable, we will just assume that a list is good enough, and treat
    # it the same as List. This condition needs to be at the end, so we don't catch other kinds
    # of Iterables with this branch.
    elif (
        origin in {collections.abc.Iterable, Iterable, List, list}
        and len(args) == 1
        and can_construct_from_params(args[0])
    ):
        value_cls = annotation.__args__[0]

        value_list = []

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return value_list

    else:
        # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
        if isinstance(popped_params, Params):
            return popped_params.as_dict()
        return popped_params


class FromParams(CustomDetHash):
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """

    @classmethod
    def from_params(
        cls: Type[T],
        params: Params,
        constructor_to_call: Callable[..., T] = None,
        constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
        **extras,
    ) -> T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses
        `FromParams` (or `Registrable`, which itself subclasses `FromParams`) gets this
        implementation for free.  If you want your class to be instantiated from params in the
        "obvious" way -- pop off parameters and hand them to your constructor with the same names --
        this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.

        The `constructor_to_call` and `constructor_to_inspect` arguments deal with a bit of
        redirection that we do.  We allow you to register particular `@classmethods` on a class as
        the constructor to use for a registered name.  This lets you, e.g., have a single
        `Vocabulary` class that can be constructed in two different ways, with different names
        registered to each constructor.  In order to handle this, we need to know not just the class
        we're trying to construct (`cls`), but also what method we should inspect to find its
        arguments (`constructor_to_inspect`), and what method to call when we're done constructing
        arguments (`constructor_to_call`).  These two methods are the same when you've used a
        `@classmethod` as your constructor, but they are `different` when you use the default
        constructor (because you inspect `__init__`, but call `cls()`).
        """

        from treetune.common.registrable import (
            Registrable,
        )  # import here to avoid circular imports

        logger.debug(
            f"instantiating class {cls} from params {getattr(params, 'params', params)} "
            f"and extras {set(extras.keys())}"
        )

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({"type": params})

        if not isinstance(params, Params):
            raise ConfigurationError(
                "from_params was passed a `params` object that was not a `Params`. This probably "
                "indicates malformed parameters in a configuration file, where something that "
                "should have been a dictionary was actually a list, or something else. "
                f"This happened when constructing an object of type {cls}."
            )

        registered_subclasses = Registrable._registry.get(cls)

        if is_base_registrable(cls) and registered_subclasses is None:
            # NOTE(mattg): There are some potential corner cases in this logic if you have nested
            # Registrable types.  We don't currently have any of those, but if we ever get them,
            # adding some logic to check `constructor_to_call` should solve the issue.  Not
            # bothering to add that unnecessary complexity for now.
            raise ConfigurationError(
                "Tried to construct an abstract Registrable base class that has no registered "
                "concrete types. This might mean that you need to use --include-package to get "
                "your concrete classes actually registered."
            )

        if registered_subclasses is not None and not constructor_to_call:
            # We know `cls` inherits from Registrable, so we'll use a cast to make mypy happy.

            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice(
                "type",
                choices=as_registrable.list_available(),
                default_to_first_choice=default_to_first_choice,
            )
            subclass, constructor_name = as_registrable.resolve_class_name(choice)
            # See the docstring for an explanation of what's going on here.
            if not constructor_name:
                constructor_to_inspect = subclass.__init__
                constructor_to_call = subclass  # type: ignore
            else:
                constructor_to_inspect = cast(
                    Callable[..., T], getattr(subclass, constructor_name)
                )
                constructor_to_call = constructor_to_inspect

            if hasattr(subclass, "from_params"):
                # We want to call subclass.from_params.
                extras = create_extras(subclass, extras)
                # mypy can't follow the typing redirection that we do, so we explicitly cast here.
                retyped_subclass = cast(Type[T], subclass)
                return retyped_subclass.from_params(
                    params=params,
                    constructor_to_call=constructor_to_call,
                    constructor_to_inspect=constructor_to_inspect,
                    **extras,
                )
            else:
                # In some rare cases, we get a registered subclass that does _not_ have a
                # from_params method (this happens with Activations, for instance, where we
                # register pytorch modules directly).  This is a bit of a hack to make those work,
                # instead of adding a `from_params` method for them somehow.  We just trust that
                # you've done the right thing in passing your parameters, and nothing else needs to
                # be recursively constructed.
                return subclass(**params)  # type: ignore
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.

            # See the docstring for an explanation of what's going on here.
            if not constructor_to_inspect:
                constructor_to_inspect = cls.__init__
            if not constructor_to_call:
                constructor_to_call = cls

            if constructor_to_inspect == object.__init__:
                # This class does not have an explicit constructor, so don't give it any kwargs.
                # Without this logic, create_kwargs will look at object.__init__ and see that
                # it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
                params.assert_empty(cls.__name__)
            else:
                # This class has a constructor, so create kwargs for it.
                constructor_to_inspect = cast(Callable[..., T], constructor_to_inspect)
                kwargs = create_kwargs(constructor_to_inspect, cls, params, **extras)

            return constructor_to_call(**kwargs)  # type: ignore

    def to_params(self) -> Params:
        """
        Returns a `Params` object that can be used with `.from_params()` to recreate an
        object just like it.

        This relies on `_to_params()`. If you need this in your custom `FromParams` class,
        override `_to_params()`, not this method.
        """

        def replace_object_with_params(o: Any) -> Any:
            if isinstance(o, FromParams):
                return o.to_params()
            elif isinstance(o, List):
                return [replace_object_with_params(i) for i in o]
            elif isinstance(o, Set):
                return {replace_object_with_params(i) for i in o}
            elif isinstance(o, Dict):
                return {
                    key: replace_object_with_params(value) for key, value in o.items()
                }
            else:
                return o

        return Params(replace_object_with_params(self._to_params()))

    def _to_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that, when turned into a `Params` object and
        then fed to `.from_params()`, will recreate this object.

        You don't need to implement this all the time. AllenNLP will let you know if you
        need it.
        """
        raise NotImplementedError()

    def det_hash_object(self) -> Any:
        return self.to_params()
