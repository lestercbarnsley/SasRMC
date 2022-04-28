#%%
import sas_rmc

CONFIG_FILE = "./data/config.yaml"

def main():
    rmc = sas_rmc.load_config(CONFIG_FILE)
    rmc.run()
    

if __name__ == "__main__":
    main()
    
#%%
