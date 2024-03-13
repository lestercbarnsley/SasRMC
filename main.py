#%%
import sys
from pathlib import Path

import sas_rmc

CONFIG_FILE = "./data/config.yaml"

def run_rmc(config_file: str) -> None:
    rmc = sas_rmc.load_config(config_file)
    rmc.run()

def run_dev_rmc() -> None:
    with open("dev_config.txt",'r') as dev_config:
        secret_dev_config = dev_config.read()
    run_rmc(secret_dev_config)

def generate_template(args: list[str]) -> None:
    command = ' '.join(args)
    sas_rmc.generate_template(command, Path(CONFIG_FILE).parent)

def main():
    argv = sys.argv
    if len(argv) <= 1 or not argv[1]:
        run_rmc(CONFIG_FILE)
    elif "dev" in argv[1]:
        run_dev_rmc()
    elif "create" in argv[1]:
        generate_template(argv[1:])
    else:
        run_rmc(CONFIG_FILE)
    

if __name__ == "__main__":
    main()
    
#%%
