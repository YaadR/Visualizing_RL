import importlib

AGENTS = [
    ("agent_Action_Value", True),
    ("agent_Policy", True),
    ("agent_Q_Lambda", False),
    ("agent_Q", True),
    ("agent_State_Value", True),
    ("agent_Value", False),
]


def main():
    print("\n", "/" * 10, "VISUALIZING RL", "\\" * 10, "\n")
    print(f"Select Agent [*=not working]")
    space = " " * 4
    for idx, (agent, status) in enumerate(AGENTS):
        name = agent.removeprefix("agent").replace("_", " ").strip()
        number = idx + 1
        working = f"" if status else f"*"
        print(f"{space}{number}: {working}{name}")

    print(f"{space}A: Run Arena")
    print(f"{space}P: Play Snake")

    selected = None
    while True:
        selected = input(" > ").strip().upper()

        if selected in "APpp":
            break

        try:
            selected = int(selected) - 1
            assert selected in range(len(AGENTS))
            break
        except:
            print("Invalid input")

    match selected:
        case "A":
            module = importlib.import_module("arena")
            module.main()
        case "P":
            module = importlib.import_module("snake_game_human")
            module.main()
        case _:
            print("/" * 10, "LOADING", "\\" * 10)
            module = importlib.import_module(AGENTS[selected][0])
            module.main()


if __name__ == "__main__":
    main()
