# Smart-Agri
this is a simple Neural Network build on Keras which takes: 
* calcium
* irrigation
* potassium
* moisture
* magnesium
* nitrogen
* phosphorus
* ph of soil
* rain
* sulphur
* temperature
* wind

as input and predicts a best 3 types of vegitables.

# Usage

edit the dataset path on the **train_model.py** file
```python
df = pd.read_csv('dataset.csv')
```

train the model
```bash
$ python train_model.py
```

test the model
```bash
$ python test_model.py
```
