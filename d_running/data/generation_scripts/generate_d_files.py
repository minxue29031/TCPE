import argparse
import os
import textwrap
import numpy as np
from datasets import load_dataset
from d_running.lib import REPLACEMENT_MARKER


def no_print(*args, **kwargs):
    pass


def dedent(code):
    code = "\n".join([line for line in code.split("\n") if len(line) > 0])
    code = textwrap.dedent(code)

    # hacky fix to circumvent inconsistent formating in test cases
    if code[0].isspace():
        indent = len(code) - len(code.lstrip())
        lines = []
        for line in code.split("\n"):
            if line[0].isspace():
                white_space = len(line) - len(line.lstrip())
                lines.append(line[min(indent, white_space):])
            else:
                lines.append(line)
        code = "\n".join(lines)

    return code


def get_test_cases(code):
    recorded_args = []

    def f_gold(*args, **kwargs):
        return 0

    def f_filled(*args):
        recorded_args.append(args)
        return 0

    original = code.replace('    ', '\\t')
    key = "if __name__ == '__main__':"
    code = code[code.find(key) + len(key):]
    code = code.replace("print", "no_print")
    code = dedent(code)

    try:
        exec(code)
    except:
        raise RuntimeError(f"Couldn't execute: \n{code}\nOriginal\n{original}")

    return recorded_args




def type_to_python_transformation(d_type):
    if "[" in d_type:
        without_brackets = d_type[:d_type.rfind("[")].strip()
        return lambda x: list(map(type_to_python_transformation(without_brackets), x))
    # {int, long, double, float, char, void, string, bool}
    if d_type in {"int", "long"}:
        restriction = np.int32
        if d_type == "long":
            restriction = np.int64
        # hacky zerodivision on overflow
        return lambda x: int(x) if x <= np.iinfo(restriction).max else 1/0
    if d_type in {"float", "double"}:
        return float
    if d_type in {"bool"}:
        return bool
    if d_type in {"string", "char"}:
        return str
    raise ValueError(f"Incompatible type {d_type}")


def get_return_type(code):
    # "parsing" using regex
    function_name = code.find("f_filled")

    full_type = code[:function_name].strip()
    is_unsigned = "unsigned" in full_type
    qualifier = "unsigned" if is_unsigned else ""

    return qualifier + full_type.split(" ")[-1]


def translate_type_java_to_d(java_type):
    special_cases = {
        "boolean": "bool",
        "integer": "int",
    }

    partial = java_type.lower().strip() 
    for orginal, replacement in special_cases.items():
        partial = partial.replace(orginal, replacement)
    return partial


def get_argument_types(code):
    arguments = code[code.find("(") + 1: code.find(")")]
    arguments = arguments.split(",")
    argument_types = []
    for argument in arguments:
        if "[" in argument:
            argument = argument.strip().split(" ")
            if argument[-1] == "]":
                argument_name_idx = argument.index("[") - 1
                argument_types.append(" ".join(argument[:argument_name_idx] + argument[argument_name_idx + 1:]))
            else:
                argument_types.append(" ".join(argument[:-1]))
        else:
            argument_types.append(" ".join(argument.strip().split(" ")[:-1]))

    return argument_types


def python_to_d(data, return_type):
    data = str(data).lower()
    return data


def main(args):
    dataset = load_dataset("path/geeks4geeks", split="train")
    dataset = dataset.filter(lambda x: x["function_java"] is not None and x["function_python"] is not None)

    for sample in dataset:
        exec(sample["function_python"].replace("print", "no_print"), globals())

        test_cases = get_test_cases(sample["testbed_python"])

        try:
            results = [f_filled(*test_case) for test_case in test_cases]
        except Exception as e:
            print(f"{sample['id']} failed with '{e}'")
            continue

        return_type = get_return_type(sample["function_java"])
        return_type = translate_type_java_to_d(return_type)

        try:
            results = str(list(map(type_to_python_transformation(return_type), results)))
        except Exception as e:
            print(f"Transformation falied with {e}")
            continue

        argument_types = get_argument_types(sample["function_java"])
        argument_types = [translate_type_java_to_d(type) for type in argument_types]

        test_cases_by_arg = [[t[i] for t in test_cases] for i in range(len(test_cases[0]))]
        try:
            test_cases_by_arg = [list(map(type_to_python_transformation(argument_type), argument)) for argument_type, argument in zip(argument_types, test_cases_by_arg)]
        except Exception as e:
            print(f"Transformation falied with {e}")
            continue

        results = f"{return_type} [ ] results = {python_to_d(results, return_type)};"
        test_cases = "\n    ".join(f"{argument_type} [ ] param{i} = {python_to_d(argument, argument_type)};"
                               for i, (argument, argument_type) in enumerate(zip(test_cases_by_arg, argument_types)))
        arguments = ", ".join(f"param{i}[i]" for i in range(len(argument_types)))

        testbed_d = D_TEMPLATE.format(results=results, test_cases=test_cases, arguments=arguments, replacement_marker=REPLACEMENT_MARKER["d"])

        with open(os.path.join(args.out_path, f"{sample['id']}.d"), "w", encoding="utf8") as f:
            f.write(testbed_d)


D_TEMPLATE = """import std.stdio;
import std.math;
import std.conv;
import std.algorithm;

{replacement_marker}

void main(){{
    {results}
    {test_cases}

    int n_success = 0;

    for (int i = 0; i < param0.length; i++) {{
       if (results[i] == f_filled({arguments})) {{
                    n_success += 1;
                }}}}

    writefln("#Results:%d, %d", n_success, param0.length);
}}
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="../d_with_params",)

    main(parser.parse_args())


