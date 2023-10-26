# Optuna dashboard

```shell
conda install optuna-dashboard # Si no esta instalado
optuna-dashboard sqlite:///multiple_studies.db
```

# COSAS PROHIBIDAS

```python
from something import *

# Please, por modularidad y sanidad mental
from something import ClaseConTodo, funcion_random

# O else, especialmente para sim utils
import simulador_utils_v02 as sim02 

sim02.funcion_random_25( ... )
```
