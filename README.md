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

run the **train_model.py** file for training
```bash
$ python tran_model.py --dataset=datasate.csv --l=learning_rate --batch=batch_size --epoch=epoch --input=input_dims --output=output_dims
```

test the model
```bash
$ python test_model.py
```
