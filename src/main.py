from config import create_config
from claude import Claude
from anthropic import Anthropic
from rich import print

if __name__ == "__main__":
    config = create_config()
    print(config)
    client = Anthropic()

    claude = Claude(client=client, model=config["claude_model"])
    claude.chat("What is 1 + 1?")
    claude.chat("Expand how you get that answer in 1 sentence?")
    print(claude.get_chat_history())
