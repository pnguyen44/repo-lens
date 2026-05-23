from config import create_config
from claude import Claude
from anthropic import Anthropic
from rich import print

if __name__ == "__main__":
    config = create_config()
    client = Anthropic()

    claude = Claude(client=client, model=config["claude_model"])

    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("quit", "exit"):
                break

            response = claude.chat(user_input)
            print(response)
    except KeyboardInterrupt:
        print("\n exiting")
