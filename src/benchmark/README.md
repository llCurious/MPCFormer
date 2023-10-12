requirements:
    pip install Crypten in editable mode: https://github.com/facebookresearch/CrypTen
    
To profile 2PC transformer, use profile_transformer.py. Set up two machines, and modify the master address accordingly.

On the first machine, run python profile_transformer.py 0, and on the second machine, run python profile_transformer.py 1. 

P.S. The lib code should be modified to enable TTPServer
`crypten/mpc/provider/ttp_provider.py`:
```python
def __init__(self):
    """Initializes a Trusted Third Party server that receives requests"""
    # Initialize connection
    crypten.init(config_file="path/to/yaml")
```

```shell
# 2PC protocol
python profile_{gpt, transformer}.py -r 0
python profile_{gpt, transformer}.py -r 1
python demo_ttp.py -r 2
```

You can play aroud with different operations by substituting value of the hidden_act and and softmax_act in the config class in the file.

- For activation, we support ["quad", "relu"]
- For softmax, we support ["softmax". "softmax_2RELU", "softmax_2QUAD"]


Additional playground:

- To profile single machine transformer, use single_transformer.py

- To profile 2PC CNN, use profile_cnn.py

