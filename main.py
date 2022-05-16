#%%
import sas_rmc

CONFIG_FILE = "./data/config.yaml"

def main():
    runner = sas_rmc.load_config(CONFIG_FILE)
    runner.run()

if __name__ == "__main__":
    main()
    #pass
    
#%%
