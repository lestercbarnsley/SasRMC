#%%


from sas_rmc import constants
from sas_rmc.acceptance_scheme import MetropolisAcceptance

rng = constants.RNG

def create_metropolis_acceptance(temperature: float, cycle: int, step: int) -> MetropolisAcceptance:
    return MetropolisAcceptance(
        temperature=temperature,
        rng_val=rng.uniform(),
        loggable_data={
            "Cycle" : cycle,
            "Step" : step
        }
    )


if __name__ == "__main__":
    print(create_metropolis_acceptance(0,0,0).rng_val)      

# %%
