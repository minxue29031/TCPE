from typing import List


def select_from_list(l: List[str]) -> int:
    for i, s in enumerate(l):
        print(f"{i+1}: {s}")
    
    user_input = None
    while True:
        user_input = input(f"1 - {len(l)}: ")
        try:
            user_input = int(user_input)
        except:
            continue
        if 0 < user_input <= len(l):
            break
    
    return user_input - 1


def user_binary_choice(message: str) -> bool:
    choice = "WRONG"
    while choice not in "yN":
        choice = input(message + " (y/N): ")
    
    return choice == "y"


def get_string_with_condition(message: str, condition) -> str:
    choice = None

    while choice is None or not condition(choice):
        choice = input(message + ": ")

    return choice