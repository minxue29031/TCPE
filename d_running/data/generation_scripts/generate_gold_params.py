import os
import json

import sys
import math
import heapq
import itertools
import collections
import numpy as np
from math import sqrt
from math import floor
from queue import Queue

sys.path.append("../..") 
from lib.translate import REPLACEMENT_MARKER


def get_gold_function(code):
    # "parsing" using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    to_fill = code.find(REPLACEMENT_MARKER["python"])
    function = code[function_start:to_fill-1]
    
    exec(function)

    return locals()["f_gold"]


def remove_gold(code, lang="java"):
    # "parsing" using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    to_fill = code.find(REPLACEMENT_MARKER["d"])
 
    return code[:function_start] + "\n" + code[to_fill:]


def get_return_type(code):
    # "parsing" using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    
    full_type = code[function_start:function_name].strip()
    is_unsigned = "unsigned" in full_type
    qualifier = "unsigned" if is_unsigned else ""

    return qualifier + full_type.split(" ")[-1]


def translate_type_java_to_d(java_type):
    partial = java_type.lower()  
    partial = "bool" if partial == "boolean" else partial

    return partial


def type_to_python_transformation(d_type):
    if d_type in {"int", "long"}:
        restriction = np.int32
        if d_type == "long":
            restriction = np.int64
        return lambda x: int(x) if x < np.iinfo(restriction).max else 1/0
    if d_type in {"float", "double"}:
        return float
    if d_type in {"bool"}:
        return bool
    if d_type in {"string", "char"}:
        return str
    raise ValueError(f"Incompatible type {d_type}")


def add_results_var(code, results, var_type):
    results = json.dumps(results)
    if var_type == "char":
        results = results.replace('"', "'")
        
    results_var = f"    {var_type}[] results = {results};\n"
    where_to_insert = code.find("void main(){\n") + len("void main(){\n")
    code = code[:where_to_insert] + results_var + code[where_to_insert:]

    return code


def edit_check(code):
    needle = """    for (int i = 0; i < param0.length; i++) {
       if ("""
    where_is_check = code.find(needle) + len(needle)
    where_is_check = code.find("f_gold(", where_is_check)
    start_of_params = code.find("f_gold(", where_is_check) + len("f_gold(")

    opening_parens = 1
    end = start_of_params
    while opening_parens > 0:
        opening_parens += bool(code[end] == "(")
        opening_parens -= bool(code[end] == ")")
        end += 1

    check = code[where_is_check:end]
    code = code[:where_is_check] + "results[i]" + code[end:]

    return code


def add_imports(code: str):
    # find last import
    last_import = code.rfind("import")
    last_import_end = code.find(";\n", last_import) + len(";")

    new_imports = """
import std.conv;
import std.algorithm;"""

    code = code[:last_import_end] + new_imports + code[last_import_end:]

    return code


def get_params(code):
    start = code.find("param = [") + len("param = ")
    opening = 1
    end = start
    while opening > 0:
        end += 1
        if code[end] == "[":
            opening += 1
        elif code[end] == "]":
            opening -= 1
    end += 1 # include "]"
    
    exec(f"params = {code[start:end]}")

    return locals()["params"]


python_path = "path_to_G4G"
d_path = "path_to_G4G"
java_path = "path_to_G4G"
d_params_path = "data/d_with_params"

for file in os.listdir(python_path):
    # skip python files without corresponding d file
    d_file_path = os.path.join(d_path, file.replace(".py", ".d"))
    java_file_path = os.path.join(java_path, file.replace(".py", ".java"))
    result_path = os.path.join(d_params_path, file.replace(".py", ".d"))
    if not os.path.isfile(d_file_path):
        continue
    if not os.path.isfile(java_file_path):
        continue

    # get return type from java function
    with open(java_file_path, "r", encoding="utf8") as f:
        java_code = f.read()
    return_type = get_return_type(java_code)
    return_type = translate_type_java_to_d(return_type)
    if return_type in {"void"}:
        continue

    python_file_path = os.path.join(python_path, file)
    with open(python_file_path, "r", encoding="utf8") as f:
        code = f.read()

    f_gold = get_gold_function(code)
    params = get_params(code)
    try:
        results = [f_gold(*param) for param in params]
    except Exception as e:
        print(f"{file} failed with '{e}'")
        continue

    with open(d_file_path, "r", encoding="utf8") as f:
        d_code = f.read()
 
    d_code = remove_gold(d_code)
    try:
        results = list(map(type_to_python_transformation(return_type), results))
    except Exception as e:
        print(f"Transformation falied with {e}")
        continue
    d_code = add_results_var(d_code, results, return_type)
    d_code = edit_check(d_code)
    d_code = add_imports(d_code)
    
    with open(result_path, "w", encoding="utf8") as f:
        f.write(d_code)
