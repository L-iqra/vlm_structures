"""
Task prompts and answer parsers for the hosted-endpoint inference client.

Task 3 is binary spalling (NSP, SP). Task 8 is damage mechanism
(COMB, FLEX, UD, SHEAR). The model is asked for a single class token.
"""

import re


SPALLING_DEFINITION = (
    "Spalling is usually defined as flakes of material that are broken off "
    "of a component or pertains to loss of cover of the component's surface. "
    "It often takes place in concrete and masonry structures, which may lead "
    "to severe consequences owing to reduction in cross section and "
    "acceleration of corrosion in reinforcing bars. Spalling can be induced "
    "by expansion force chemically or mechanically."
)

TASK3_PROMPT = (
    "You are a structural engineer tasked with classifying damage types in civil engineering structures.\n"
    f"Definition of spalling: {SPALLING_DEFINITION}\n\n"
    "Given the image, decide if spalling is present.\n"
    "Respond with exactly one token:\n"
    "- SP (spalling)\n"
    "- NSP (non-spalling)\n"
    "Do not output anything else."
)

DAMAGE_TYPE_DEFINITION = (
    "Damage type refers to a complex and abstract semantic vision pattern, "
    "defined as flexural (FLEX), shear (SHEAR), and combined (COMB) damage. "
    "These damage types have direct implications on the mechanical properties "
    "and seismic design. Analogous to failure types introduced in Moehle (2014) "
    "for seismic loading, the definitions are as follows: "
    "1) If most cracks occur in the horizontal or vertical direction, or at the "
    "end of a component with vertical or horizontal edges, it is flexural-type "
    "damage (FLEX), which is more ductile. "
    "2) If most cracks occur in a diagonal direction or form an X or V pattern, "
    "it is shear-type damage (SHEAR), which is more brittle. "
    "3) If the distribution of cracks is irregular or accompanied with heavy "
    "spalling, it is combined-type damage (COMB). "
    "4) If damage is not visible, classify as undamaged state (UD)."
)

TASK8_PROMPT = (
    "You are a structural damage inspection assistant.\n"
    f"Definition of damage type: {DAMAGE_TYPE_DEFINITION}\n\n"
    "Given the image, classify the damage type.\n"
    "Respond with exactly one token:\n"
    "- COMB (combined damage)\n"
    "- FLEX (flexural damage)\n"
    "- UD (undamaged state)\n"
    "- SHEAR (shear damage)\n"
    "Do not output anything else."
)

REFUSAL_PATTERNS = (
    r"\bI'?M SORRY\b",
    r"\bI CANNOT\b",
    r"\bI CAN'?T\b",
    r"\bUNABLE TO\b",
    r"\bAS AN AI\b",
    r"\bCANNOT DETERMINE\b",
    r"\bNOT POSSIBLE TO\b",
)


def parse_spalling_prediction(raw_text):
    text = raw_text.strip().upper()
    if re.search(r"\bNSP\b", text):
        return 0, "NSP"
    if re.search(r"\bSP\b", text):
        return 1, "SP"
    return None, None


def parse_damage_prediction(raw_text):
    text = raw_text.strip().upper()

    if re.search(r"\bSHEAR\b", text):
        return 3, "SHEAR"
    if re.search(r"\bFLEX\b", text) or re.search(r"\bFLEXURAL\b", text):
        return 1, "FLEX"
    if re.search(r"\bCOMB\b", text) or re.search(r"\bCOMBINED\b", text):
        return 0, "COMB"
    if re.search(r"\bUD\b", text) or re.search(r"\bUNDAMAGED\b", text):
        return 2, "UD"

    return None, None


def classify_parse_status(raw_text, pred_label):
    if pred_label is not None:
        return "ok"
    text = (raw_text or "").strip().upper()
    if not text:
        return "empty"
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text):
            return "refusal"
    return "unparseable"


class TaskConfig:
    def __init__(self, task_id, short_name, title, label_id_to_name, prompt, parser, max_tokens):
        self.task_id = task_id
        self.short_name = short_name
        self.title = title
        self.label_id_to_name = label_id_to_name
        self.prompt = prompt
        self.parser = parser
        self.max_tokens = max_tokens

    @property
    def label_ids(self):
        return tuple(sorted(self.label_id_to_name))

    @property
    def label_names(self):
        return tuple(self.label_id_to_name[i] for i in self.label_ids)

    def name_for(self, label_id):
        return self.label_id_to_name.get(label_id, f"UNKNOWN_{label_id}")


TASKS = {
    3: TaskConfig(
        task_id=3,
        short_name="spalling",
        title="Task 3: Spalling condition",
        label_id_to_name={0: "NSP", 1: "SP"},
        prompt=TASK3_PROMPT,
        parser=parse_spalling_prediction,
        max_tokens=8,
    ),
    8: TaskConfig(
        task_id=8,
        short_name="damage_type",
        title="Task 8: Damage type",
        label_id_to_name={0: "COMB", 1: "FLEX", 2: "UD", 3: "SHEAR"},
        prompt=TASK8_PROMPT,
        parser=parse_damage_prediction,
        max_tokens=8,
    ),
}


def get_task(task_id):
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id {task_id}; expected one of {sorted(TASKS)}")
    return TASKS[task_id]
