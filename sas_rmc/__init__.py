from .rmc_runner import RmcRunner

def load_config(config_file: str) -> RmcRunner:
    return RmcRunner(config_file)
    
if __name__ == "__main__":
    pass
    
# %%