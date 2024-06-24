# OptTomo

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
