from inference_full_model import load_pipeline, ask
from colorama import Fore, Style, init

init(autoreset=True)

LOG_FILE = "chat_log.txt"

if __name__ == "__main__":
    print(Fore.CYAN + "MedAssist-QA (not medical advice). Type 'exit' to quit.")
    gen = load_pipeline()
    while True:
        q = input(Fore.GREEN + "\nYou: " + Style.RESET_ALL).strip()
        if q.lower() in {"exit", "quit"}:
            print(Fore.YELLOW + "Goodbye!")
            break
        answer = ask(gen, q)
        print(Fore.MAGENTA + "Bot:" + Style.RESET_ALL, answer)

        # --- Save to log file ---
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"You: {q}\nBot: {answer}\n\n")
