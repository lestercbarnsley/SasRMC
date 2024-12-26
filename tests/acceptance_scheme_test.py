#%%

from pytest import mark, raises as pytest_raises

from sas_rmc.acceptance_scheme import MetropolisAcceptance, AcceptanceEarlyTemination

@mark.parametrize(
        ["old_goodness_of_fit", "new_goodness_of_fit", "temperature","acceptable"],
        [
            (21, 20, 0, True),
            (20, 21, 0, False),
            (20, 21, 1, True)
        ]
)
def test_metropolis_acceptance_is_acceptable(old_goodness_of_fit, new_goodness_of_fit, temperature, acceptable):
    assert acceptable == MetropolisAcceptance(temperature=temperature,rng_val=0).is_acceptable(old_goodness_of_fit, new_goodness_of_fit)

def test_acceptance_early_acceptance():
    acceptance = AcceptanceEarlyTemination(MetropolisAcceptance(0, 0))
    with pytest_raises(StopIteration):
        acceptance.is_acceptable(0, 0)

def test_early_acceptance_passes():
    metropilis_acceptance = MetropolisAcceptance(0, 0)
    acceptance = AcceptanceEarlyTemination(metropilis_acceptance)
    old_goodness_of_fit = 2
    new_goodness_of_fit = 2
    assert acceptance.is_acceptable(old_goodness_of_fit, new_goodness_of_fit) == acceptance.acceptance_scheme.is_acceptable(old_goodness_of_fit, new_goodness_of_fit)

def test_has_loggable_data():
    metropilis_acceptance = MetropolisAcceptance(0, 0)
    acceptance = AcceptanceEarlyTemination(metropilis_acceptance)
    assert isinstance(acceptance.get_loggable_data(), dict)


