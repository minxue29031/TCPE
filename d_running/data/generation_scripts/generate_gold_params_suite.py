import os


def get_gold_function(code):
    # parsing using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    to_fill = code.find("//TOFILL")
    function = code[function_start:to_fill-1]

    return function

originals_path = "data/transcoder_evaluation_gfg/d"
d_params_path = "data/d_with_params"
outs_path = "code_under_test"
for file in os.listdir(d_params_path):
    original_path = os.path.join(originals_path, file)
    d_path = os.path.join(d_params_path, file)
    with open(original_path, "r", encoding="utf8") as f:
        code = f.read()
    
    gold_function = get_gold_function(code)
    gold_function = gold_function.replace("f_gold", "f_filled")

    with open(d_path, "r", encoding="utf8") as f:
        code = f.read()
    code = code.replace("//TOFILL", gold_function)
    
    out_path = os.path.join(outs_path, file)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(code)