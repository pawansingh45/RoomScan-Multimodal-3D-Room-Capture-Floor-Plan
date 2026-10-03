"""Minimal JSON-Schema (draft-07 subset) validator so the output contract is enforced without extra deps."""
import json, os

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema", "roomscan.plan.schema.json")


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


_T = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(v, t):
    ts = t if isinstance(t, list) else [t]
    for x in ts:
        if x == "number" and isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if x == "integer" and isinstance(v, int) and not isinstance(v, bool):
            return True
        if x in _T and isinstance(v, _T[x]):
            return True
    return False


def _resolve(ref, root):
    node = root
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def validate(inst, schema, root=None, path="$"):
    root = root or schema
    errs = []
    if "$ref" in schema:
        return validate(inst, _resolve(schema["$ref"], root), root, path)
    if "type" in schema and not _type_ok(inst, schema["type"]):
        return [f"{path}: expected {schema['type']}, got {type(inst).__name__}"]
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in {schema['enum']}")
    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            errs.append(f"{path}: {inst} < minimum {schema['minimum']}")
        if "maximum" in schema and inst > schema["maximum"]:
            errs.append(f"{path}: {inst} > maximum {schema['maximum']}")
    if isinstance(inst, dict):
        for k in schema.get("required", []):
            if k not in inst:
                errs.append(f"{path}: missing required '{k}'")
        for k, sub in schema.get("properties", {}).items():
            if k in inst:
                errs += validate(inst[k], sub, root, f"{path}.{k}")
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, x in enumerate(inst):
                errs += validate(x, schema["items"], root, f"{path}[{i}]")
    return errs


def validate_plan(plan):
    return validate(plan, load_schema())
