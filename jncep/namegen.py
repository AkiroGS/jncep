from collections import namedtuple
from collections.abc import Iterable
import copy
import logging
import sys
from typing import List

from attr import define

from .utils import to_safe_filename, to_safe_foldername

logger = logging.getLogger(__name__)

GEN_RULES = [
    # "slug",
    # "volume_part_2_digit",
    # "volume_dot_volume_part",
    # "series_part_dot_volume",  # Volume 2.5
    # "volume_dot_part",  # only for single parts : Part 2.5 or Part 5.2.4
    # "zero_padding_series_part",
    # "zero_padding_volume",
    # "zero_padding_volume_part",
    # "zero_padding_part",
    # "no_part_list",
    # "no_small_words",
    # "acronym",
    # "first_letters",
    # "no_separator",
    # # "unsafe"
    # "safe",  # default : unsafe must be explicitly specified
    ######
    "fc_rm",
    "fc_short",
    "fc_full",
    "p_split_part",
    "v_split_volume",
    "to_volume",
    "to_series",
    "p_full",
    "pn_rm",
    "pn_rm_if_complete",
    "pn_prepend_vn_if_multiple",
    "pn_prepend_vn",
    "pn_0pad",
    "pn_short",
    "pn_full",
    "v_full",
    "vn_rm",
    "vn_number",
    "vn_merge",
    "vn_0pad",
    "vn_pn_merge",
    "vn_short",
    "vn_full",
    "s_full",
    "s_slug",
    "s_rm_stopwords",
    "s_acronym",
    "s_trigram",
    "_t",  # special processing : takes title as starting point
    "text",
    "max_len60",
    "rm_space",
    "filesafe_underscore",
    "filesafe_space",
    "foldersafe_underscore",
    "foldersafe_space",
]

DEF_SEP = "|"
RULE_SEP = ">"

RULE_SPECIAL_PREVIOUS = "_t"

TITLE_SECTION = "t"
FILENAME_SECTION = "n"
# folder instead of directory since the option to output into folder is --folder
FOLDER_SECTION = "f"

DEFAULT_NAMEGEN_RULES = (
    "t:fc_full>p_full>pn_rm_if_complete>pn_prepend_vn_if_multiple>pn_full>v_full>"
    + "vn_full>s_full>text"
    + "|n:_t>filesafe_underscore"
    + "|f:to_series>pn_rm>vn_rm>text>foldersafe_underscore"
)

CACHED_PARSED_NAMEGEGEN_RULES = None

# TODO use enum
FC_COM = "FC"
PN_COM = "P_NUM"
VN_COM = "V_NUM"
P_COM = "P_NAME"
V_COM = "V_NAME"
S_COM = "S_NAME"
STR_COM = "STR"


@define
class Component:
    tag: str
    value: object
    transformed_value: object = None
    output: str = None


FC = namedtuple("FC", "final complete")


class InvalidNamegenRulesError(Exception):
    pass


def _init_module():
    # TODO init Gen_Rules
    pass


def parse_namegen_rules(namegen_rules):
    global CACHED_PARSED_NAMEGEGEN_RULES

    if CACHED_PARSED_NAMEGEGEN_RULES:
        return CACHED_PARSED_NAMEGEGEN_RULES

    default_gen_rules = _do_parse_namegen_rules(DEFAULT_NAMEGEN_RULES)

    if namegen_rules:
        # always the same during the execution so read once and cache
        gen_rules = _do_parse_namegen_rules(namegen_rules)

        if TITLE_SECTION not in gen_rules:
            gen_rules[TITLE_SECTION] = default_gen_rules.title
        if FILENAME_SECTION not in gen_rules:
            gen_rules[FILENAME_SECTION] = default_gen_rules.filename
        if FOLDER_SECTION not in gen_rules:
            gen_rules[FOLDER_SECTION] = default_gen_rules.folder

        CACHED_PARSED_NAMEGEGEN_RULES = gen_rules
    else:
        CACHED_PARSED_NAMEGEGEN_RULES = default_gen_rules

    return CACHED_PARSED_NAMEGEGEN_RULES


def _do_parse_namegen_rules(namegen_rules):
    try:
        tnf = namegen_rules.split(DEF_SEP)
        tnf_defs = _extract_components(tnf)
        if logger.level <= logging.DEBUG:
            logger.debug(f"tnf = {tnf_defs}")

        gen_rules = {}
        for key, gen_definition in tnf_defs.items():
            if gen_definition:
                rules = _validate(gen_definition.split(RULE_SEP))
                gen_rules[key] = rules

        return gen_rules

    except Exception as ex:
        # TODO more precise feedback
        raise InvalidNamegenRulesError(f"Invalid: {namegen_rules}") from ex


def _extract_components(tnf):
    tnf_defs = {}
    for component in tnf:
        for c_def in [TITLE_SECTION, FILENAME_SECTION, FOLDER_SECTION]:
            if component.startswith(c_def + ":"):
                tnf_defs[c_def] = component[2:]
    return tnf_defs


def _validate(arr):
    rules = []
    for c in arr:
        c = c.strip()
        if c not in GEN_RULES:
            raise InvalidNamegenRulesError(f"Invalid rule: {c}")
        rules.append(c)
    return rules


def generate_names(series, volumes, parts, fc, parsed_namegen_rules):
    outputs = []
    for section in [TITLE_SECTION, FILENAME_SECTION, FOLDER_SECTION]:
        rules = parsed_namegen_rules[section]

        # special processing
        if rules[0] == RULE_SPECIAL_PREVIOUS:
            previous = outputs[-1]
            # clone component since it will be modified
            components = [copy.copy(previous)]
            rules = rules[1:]
        else:
            # initilize new for each section since modified
            components = _initialize_components(series, volumes, parts, fc)

        # single STR_COM
        _apply_rules(components, rules, outputs)

        if len(components) != 1 or components[0].tag != STR_COM:
            logger.debug(f"Components : {components}")
            raise InvalidNamegenRulesError(
                "Invalid namegen definition: should generate a string"
            )

        outputs.append(components[0])

    return [o.output for o in outputs]


def _initialize_components(series, volumes, parts, fc):
    # TODO initilize series, volume, part structs specific to this processing
    # to handle split, pad, merge
    # DONE ? check
    if len(parts) == 1:
        components = [Component(P_COM, parts[0])]
    else:
        if len(volumes) > 1:
            components = [
                Component(S_COM, series),
                Component(VN_COM, volumes, _default_vn(volumes)),
                Component(PN_COM, parts, _default_pn(parts)),
            ]
        else:
            components = [
                Component(V_COM, volumes[0]),
                Component(PN_COM, parts, _default_pn(parts)),
            ]

    components.append(Component(FC_COM, fc))

    return components


def _apply_rules(components: List[Component], rules, outputs):
    for rule in rules:
        f_rule = getattr(sys.modules[__name__], rule, None)
        # array modified in place
        f_rule(components)


def fc_full(components: List[Component]):
    component = _find_component_type(FC_COM, components)
    if not component:
        return
    # priority on complete
    if component.value.complete:
        component.output = "[Complete]"
    elif component.value.final:
        component.output = "[Final]"


def p_full(components: List[Component]):
    component = _find_component_type(P_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


def v_full(components: List[Component]):
    component = _find_component_type(V_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


def s_full(components: List[Component]):
    component = _find_component_type(S_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


def pn_full(components: List[Component]):
    component = _find_component_type(PN_COM, components)
    if not component:
        return

    part_numbers = component.transformed_value
    if len(part_numbers) > 1:
        part0 = part_numbers[0]
        part1 = part_numbers[-1]
        component.output = f"[Parts {part0} to {part1}]"
    else:
        component.output = f"Part {part_numbers[0]}"


def vn_full(components: List[Component]):
    component = _find_component_type(VN_COM, components)
    if not component:
        return

    volumes = component.value
    if len(volumes) > 1:
        volume_nums = [volume.num for volume in volumes]
        volume_nums = sorted(volume_nums)
        volume_nums = [str(vn) for vn in volume_nums]
        volume_nums = ", ".join(volume_nums[:-1]) + " & " + volume_nums[-1]
        component.output = f"Volumes {volume_nums}"
    else:
        volume = volumes[0]
        component.output = f"Volume {volume.num} "


def to_series(components: List[Component]):
    v_component = _find_component_type(V_COM, components)
    if v_component:
        volume = v_component.value
        vn_component = Component(VN_COM, [volume], _default_vn([volume]))
        series_component = Component(S_COM, volume.series)
        _replace_component(components, v_component, series_component, vn_component)

    p_component = _find_component_type(P_COM, components)
    if p_component:
        part = p_component.value
        volume = part.volume
        pn_component = Component(PN_COM, [part], _default_pn([part]))
        vn_component = Component(VN_COM, [volume], _default_vn([volume]))
        series_component = Component(S_COM, volume.series)
        _replace_component(
            components, p_component, series_component, vn_component, pn_component
        )


def _default_vn(volumes):
    volume_numbers = [str(v.num) for v in volumes]
    return volume_numbers


def _default_pn(parts):
    part_numbers = [str(p.num_in_volume) for p in parts]
    return part_numbers


def pn_rm(components: List[Component]):
    pn_component = _find_component_type(PN_COM, components)
    if not pn_component:
        return
    _del_component(components, pn_component)


def pn_rm_if_complete(components: List[Component]):
    pn_component = _find_component_type(PN_COM, components)
    if not pn_component:
        return
    fc_component = _find_component_type(FC_COM, components)
    if not fc_component:
        return

    if fc_component.value.complete:
        _del_component(components, pn_component)


def pn_prepend_vn_if_multiple(components: List[Component]):
    pn_component = _find_component_type(PN_COM, components)
    if not pn_component:
        return
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return

    if len(vn_component.value) == 1:
        return

    parts = pn_component.value
    volumes = vn_component.value
    for i, part in enumerate(parts):
        tr_pn = pn_component.transformed_value[i]
        volume = part.volume
        volume_i = volumes.index(volume)
        tr_vn = vn_component.transformed_value[volume_i]
        new_tr_pn = f"{tr_vn}.{tr_pn}"
        pn_component.transformed_value[i] = new_tr_pn


def vn_rm(components: List[Component]):
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return
    _del_component(components, vn_component)


def filesafe_underscore(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    # TODO create new insteaf of updating ?
    str_component.output = to_safe_filename(str_component.output, "_")


def filesafe_space(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_filename(str_component.output, " ")


def foldersafe_underscore(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_foldername(str_component.output, "_")


def foldersafe_space(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_foldername(str_component.output, " ")


def max_len60(components: List[Component]):
    pass


def text(components: List[Component]):
    outputs = [c.output for c in components if c.output]
    str_value = " ".join(outputs)
    str_component = Component(STR_COM, None, output=str_value)
    _replace_all(components, str_component)


def _find_component_type(ctype, components: List[Component]):
    for component in components:
        if component.tag == ctype:
            # supposes only one
            return component

    return None


def _find_str_component_implicit_text(components):
    str_component = _find_component_type(STR_COM, components)
    if not str_component:
        # implicit
        text(components)
        str_component = _find_component_type(STR_COM, components)

    return str_component


def _is_iterable(value):
    return isinstance(value, Iterable)


def _replace_component(components, item, *items):
    item_index = _index(components, item)
    components[item_index : item_index + 1] = items


def _del_component(components, item):
    item_index = _index(components, item)
    del components[item_index]


def _replace_all(components, *items):
    components[:] = items


def _index(components, item):
    item_index = next(
        (i for i, component in enumerate(components) if component is item), None
    )
    return item_index


_init_module()
