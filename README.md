# salto

> Playing with embedding vectors


## Installation

For the moment, you can install this library using
```bash
pip install git+https://github.com/krzjoa/salto.git
```

## Motivation

The goal of the **salto** package is to explore embeddings and check, 
how the distance between two points (vectors) can be interpreted.
We get two arbitrary selected points, such as embedding vectors for **ice** and **fire**
draw a straight line passing trough the both these points. Then, we treat the 
newly created line as a new axis by projecting the rest of the points onto this line.

 

<img src = "https://raw.githubusercontent.com/krzjoa/salto/main/examples/plot_3.png"></a>
<center> <i>Drawn using: <a>https://www.geogebra.org/m/JMMKv7cx<a></i>
 
I named the package **salto**, which means *somersault* in many languages or simply *jump* in Romance languages like Italian, where this word originally comes from.
It's because the operation of changing space for me resembles a kind of acrobatics 😉.

## Usage

```python
import numpy as np
import spacy
import salto

nlp = spacy.load('en_core_web_md')

fire = nlp('fire')
ice = nlp('ice')

ice_fire_axis = salto.axis(ice.vector, fire.vector)

cold = ['ice cream', 'polar', 'snow', 'winter', 'fridge', 'Antarctica']
warm = ['boiling water', 'tropical', 'sun', 'summer', 'oven', 'Africa']

cold_vecs = [nlp(w).vector for w in cold]
warm_vecs = [nlp(w).vector for w in warm]

cold_values = [ice_fire_axis(p) for p in cold_vecs]
warm_values = [ice_fire_axis(p) for p in warm_vecs]

ice_fire_axis.plot(
        {'values': cold_values, 'labels': cold, 'color': 'tab:blue'},
        {'values': warm_values, 'labels': warm, 'color': 'tab:red'},
        poles = {'negative': {'label': 'ice', 'color': 'blue'}, 
                 'positive': {'label': 'fire', 'color': 'red'}}
    )  
```
<img src = "https://raw.githubusercontent.com/krzjoa/salto/main/examples/word-embedding_45_0.png"></a>