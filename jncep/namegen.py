from collections import namedtuple
import copy
import logging
import numbers
import sys
from typing import List

from attr import define

from .utils import getConsole, to_safe_filename, to_safe_foldername

logger = logging.getLogger(__name__)
console = getConsole()

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
    "p_to_volume",
    "p_to_series",
    "p_split_part",
    "p_title",
    "pn_rm",
    "pn_rm_if_complete",
    "pn_prepend_vn_if_multiple",
    "pn_prepend_vn",
    "pn_0pad",
    "pn_short",
    "pn_full",
    "v_to_series",
    "v_split_volume",
    "v_title",
    "vn_rm",
    "vn_number",
    "vn_merge",
    "vn_0pad",
    "vn_short",
    "vn_full",
    "to_series",
    "s_title",
    "s_slug",
    "s_rm_stopwords",
    "s_acronym",
    "s_trigram",
    "s_max_len60",
    "_t",  # special processing : takes title as starting point
    "text",
    "rm_space",
    "filesafe_underscore",
    "filesafe_space",
    # TODO remove ? use only filesafe
    "foldersafe_underscore",
    "foldersafe_space",
]

DEF_SEP = "|"
RULE_SEP = ">"

RULE_SPECIAL_PREVIOUS = "_t"

TITLE_SECTION = "t"
FILENAME_SECTION = "n"
# folder instead of directory since the option to output into folder is --subfolder
FOLDER_SECTION = "f"

DEFAULT_NAMEGEN_RULES = (
    "t:fc_full>p_title>pn_rm_if_complete>pn_prepend_vn_if_multiple>pn_full>v_title>"
    + "vn_full>s_title>text"
    + "|n:_t>filesafe_underscore"
    + "|f:to_series>fc_rm>pn_rm>vn_rm>s_title>text>foldersafe_underscore"
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

VN_INTERNAL = "VN_INTERNAL"
VN_MERGED = "VN_MERGED"


# TODO for JNC Nina : something different => + i18n of PArt, Volume
# should be enough
EN_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


@define
class Component:
    tag: str
    value: object
    # TODO for PN , VN => reverse : change to base_value
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

        components, rules = _initialize_components(
            series, volumes, parts, fc, rules, outputs
        )

        _apply_rules(components, rules, outputs)

        if len(components) != 1 or components[0].tag != STR_COM:
            logger.debug(f"Components : {components}")
            raise InvalidNamegenRulesError(
                "Invalid namegen definition: should generate a string"
            )

        outputs.append(components[0])

    return [o.output for o in outputs]


def _initialize_components(series, volumes, parts, fc, rules, outputs):
    # special processing
    if rules[0] == RULE_SPECIAL_PREVIOUS:
        previous = outputs[-1]
        # clone component since it will be modified
        components = [copy.copy(previous)]
        rules = rules[1:]
    else:
        # initilize new for each section since modified
        components = _default_initialize_components(series, volumes, parts, fc)

    return components, rules


def _default_initialize_components(series, volumes, parts, fc):
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
        logger.debug(f"Apply rule: {rule}")
        # array modified in place
        f_rule(components)


def fc_rm(components: List[Component]):
    component = _find_component_type(FC_COM, components)
    if not component:
        return
    _del_component(components, component)


def fc_short(components: List[Component]):
    component = _find_component_type(FC_COM, components)
    if not component:
        return
    if component.value.complete:
        component.output = "[C]"
    elif component.value.final:
        component.output = "[F]"


def fc_full(components: List[Component]):
    component = _find_component_type(FC_COM, components)
    if not component:
        return
    # priority on complete
    if component.value.complete:
        component.output = "[Complete]"
    elif component.value.final:
        component.output = "[Final]"


def p_to_volume(components: List[Component]):
    component = _find_component_type(P_COM, components)
    if not component:
        return
    part = component.value
    volume = part.volume
    v_com = Component(V_COM, volume)
    pn_com = Component(PN_COM, [part], _default_pn([part]))
    _replace_component(components, component, v_com, pn_com)


def p_to_series(components: List[Component]):
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


def p_split_part(components: List[Component]):
    raise NotImplementedError()


def p_title(components: List[Component]):
    component = _find_component_type(P_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


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
        tr_vn = _vn_to_single(vn_component.transformed_value[volume_i])
        new_tr_pn = f"{tr_vn}.{tr_pn}"
        pn_component.transformed_value[i] = new_tr_pn


def pn_prepend_vn(components: List[Component]):
    pn_component = _find_component_type(PN_COM, components)
    if not pn_component:
        return
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return

    parts = pn_component.value
    volumes = vn_component.value
    for i, part in enumerate(parts):
        tr_pn = pn_component.transformed_value[i]
        volume = part.volume
        volume_i = volumes.index(volume)
        tr_vn = _vn_to_single(vn_component.transformed_value[volume_i])
        new_tr_pn = f"{tr_vn}.{tr_pn}"
        pn_component.transformed_value[i] = new_tr_pn


def pn_0pad(components: List[Component]):
    component = _find_component_type(PN_COM, components)
    if not component:
        return
    part_numbers = component.transformed_value
    component.transformed_value = [str(pn).zfill(2) for pn in part_numbers]


def pn_short(components: List[Component]):
    component = _find_component_type(PN_COM, components)
    if not component:
        return

    part_numbers = component.transformed_value
    if len(part_numbers) > 1:
        part0 = part_numbers[0]
        part1 = part_numbers[-1]
        component.output = f"{part0}-{part1}"
    else:
        component.output = f"{part_numbers[0]}"


def pn_full(components: List[Component]):
    component = _find_component_type(PN_COM, components)
    if not component:
        return

    part_numbers = component.transformed_value
    if len(part_numbers) > 1:
        part0 = part_numbers[0]
        part1 = part_numbers[-1]
        component.output = f"Parts {part0} to {part1}"
    else:
        component.output = f"Part {part_numbers[0]}"


def v_to_series(components: List[Component]):
    component = _find_component_type(V_COM, components)
    if not component:
        return
    volume = component.value
    series = volume.series
    s_com = Component(S_COM, series)
    vn_com = Component(VN_COM, [volume], _default_vn([volume]))
    _replace_component(components, component, s_com, vn_com)


def v_split_volume(components: List[Component]):
    raise NotImplementedError()


def v_title(components: List[Component]):
    component = _find_component_type(V_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


def vn_rm(components: List[Component]):
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return
    _del_component(components, vn_component)


def vn_number(components: List[Component]):
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return

    volume_numbers = vn_component.transformed_value
    # for voumes with 2 parts : like AoaB : Part 5 Volume 2
    # or Volume 3 Part Four
    for v in volume_numbers:
        # TODO specific type : CompoundVN
        # => add when implementing v_split_volume
        if _is_list(v):
            for j, p in enumerate(v):
                if isinstance(p[0], str):
                    # TODO for JNC Nina : something diff
                    p0l = p[0].lower()
                    if p0l in EN_NUMBERS:
                        n = EN_NUMBERS[p0l]
                        v[j] = (n, p[1])


def vn_merge(components: List[Component]):
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return

    volume_numbers = vn_component.transformed_value
    for i, v in enumerate(volume_numbers):
        if _is_list(v):
            # otherwise no need to merge
            to_merge = [p[0] for p in v]
            volume_numbers[i] = (".".join(to_merge), VN_MERGED)


def vn_0pad(components: List[Component]):
    vn_component = _find_component_type(VN_COM, components)
    if not vn_component:
        return

    volume_numbers = vn_component.transformed_value
    for i, v in enumerate(volume_numbers):
        # TODO always store VN as a list of list of tuples
        if _is_list(v):
            # TODO do instead: int(...). + format ?
            padded = [(str(p[0]).zfill(2), p[1]) for p in v]
            volume_numbers[i] = padded
        else:
            volume_numbers[i] = (str(v[0]).zfill(2), v[1])


def vn_short(components: List[Component]):
    component = _find_component_type(VN_COM, components)
    if not component:
        return

    # implicit
    vn_merge(components)

    volumes = component.transformed_value
    if len(volumes) > 1:
        volume0 = volumes[0][0]
        volume1 = volumes[1][0]
        volume_nums = f"{volume0}-{volume1}"
        component.output = f"{volume_nums}"
    else:
        volume = volumes[0][0]
        component.output = f"{volume.num} "


def vn_full(components: List[Component]):
    component = _find_component_type(VN_COM, components)
    if not component:
        return

    volumes = component.transformed_value
    if len(volumes) > 1:
        # implicit
        vn_merge(components)

        volume_nums = [volume[0] for volume in volumes]
        volume_nums = [str(vn) for vn in volume_nums]
        volume_nums = ", ".join(volume_nums[:-1]) + " & " + volume_nums[-1]
        component.output = f"Volumes {volume_nums}"
    else:
        # in this case : the 2 part Volume number may have been preserved
        volume = volumes[0]
        if _is_list(volume) and len(volume) > 1:
            # [(2, "Part"), (5, "Volume")]
            component.output = " ".join([f"{p[1]} {p[0]}" for p in volume])
        else:
            # (2, "VN_INTERNAL")
            component.output = f"Volume {volume[0]}"


def to_series(components: List[Component]):
    v_to_series(components)
    p_to_series(components)


def s_title(components: List[Component]):
    component = _find_component_type(S_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.title


def s_slug(components: List[Component]):
    component = _find_component_type(S_COM, components)
    if not component:
        return
    component.output = component.value.raw_data.slug


def s_rm_stopwords(components: List[Component]):
    raise NotImplementedError()


def s_acronym(components: List[Component]):
    raise NotImplementedError()


def s_trigram(components: List[Component]):
    raise NotImplementedError()


def s_max_len60(components: List[Component]):
    raise NotImplementedError()


def text(components: List[Component]):
    outputs = [c.output for c in components if c.output]
    str_value = " ".join(outputs)
    str_component = Component(STR_COM, None, output=str_value)
    _replace_all(components, str_component)


def rm_space(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = str_component.output.replace(" ", "")


def filesafe_underscore(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    # TODO create new insteaf of updating ?
    str_component.output = to_safe_filename(str_component.output, "_")


def filesafe_space(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_filename(str_component.output, " ")


def foldersafe_underscore(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_filename(str_component.output, "_")


def foldersafe_space(components: List[Component]):
    str_component = _find_str_component_implicit_text(components)
    str_component.output = to_safe_filename(str_component.output, " ")


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


def _default_vn(volumes):
    volume_numbers = [(v.num, VN_INTERNAL) for v in volumes]
    return volume_numbers


def _default_pn(parts):
    part_numbers = [p.num_in_volume for p in parts]
    return part_numbers


def _vn_to_single(vn):
    if _is_list(vn):
        items = [p[0] for p in vn]
        return ".".join(items)
    return vn[0]


def _is_number(v):
    return isinstance(v, numbers.Number)


def _is_list(v):
    return isinstance(v, list)


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
