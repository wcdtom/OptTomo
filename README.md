# OptTomo

<div style="text-align: center;">
    <img src="./Results/gamma_theory.png" alt="plot" style="width: 80%;"/>
</div>

* run ./main_multiprocess.sh to concurrently call tomo_fiber.py with different random seeds.
* ./tomo_fiber.py solve the Linear Least Squares Estimation of longitudinal power profile based on the simulation data
provided by [OpticCommPy](https://github.com/edsonportosilva/OptiCommPy).
* set the parameters related to transmitter in ./signal_generator_coherent.py
* set the parameters related to fiber and tomography in ./tomo_fiber.py.

### Requirements

If you use the Pycharm virtual environment (else pass) :

```
source .venv/bin/activate
```

Install Dependencies from ./requirements.txt

```
pip install -r requirements.txt
```

### TODO
1. Reproducing the results published by NTT (perfect receiver, regardless of DSP algorithms).
2. Introducing Error and corresponding DSP calibration algorithms.
3. Integrating the tomography approach with the [GNpy](https://github.com/Telecominfraproject/oopt-gnpy) library.
4. Expanding our model, e.g., to the scenario considering polarization multiplexing, space division multiplexing, or SRS.

### Computational Overhead
* $\gamma = (Re\[GG\])^{-1}$Re\[GA_1\]$
* For the scenario with a pilot signal, the burden of computation can be performed
in advance.
