import importlib

AGENTS = [
    "agent_Action_Value",
    "agent_Policy",
    "agent_Q_Lambda",
    "agent_Q",
    "agent_State_Value",
    "agent_Value",
]


def main():
    print("Select Agent")
    for idx, agent in enumerate(AGENTS):
        print(
            f"{' '*4}{idx+1}: {agent.removeprefix('agent').replace('_', ' ').strip()}"
        )

    selected = input()
    module = importlib.import_module(AGENTS[int(selected) - 1])
    module.main()


if __name__ == "__main__":
    main()
