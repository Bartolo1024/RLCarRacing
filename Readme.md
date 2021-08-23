### DQN CarRacing-v0

## Setup

* Please install anaconda package manager.
* Create environment: `conda env create -f environment.yaml
* Activate environment with the command: `conda activate rl`
* Please specify your neptune project in the: ```configs/base_config.yaml``` or delete the row if you want to use screen logging.
* Run example experiment with configuration:
```python train.py configs/base_config.yaml```
  
## Results

Result weights are store in the `output` directory.
