import os


def fill_gold(code):
    # parsing using regex
    function_name = code.find("f_gold")
    function_start = code.rfind("\n", 0, function_name)
    to_fill = code.find("//TOFILL")
    function = code[function_start:to_fill-1]
    function = function.replace("f_gold", "f_filled")
    code = code.replace("//TOFILL", function)

    return code

originals_path = "data/transcoder_evaluation_gfg/d"
outs_path = "code_under_test"
for file in os.listdir(originals_path):
    original_path = os.path.join(originals_path, file)
    with open(original_path, "r", encoding="utf8") as f:
        code = f.read()
    out_path = os.path.join(outs_path, file)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(fill_gold(code))