# 1 - Look at the big picture

#### 1 - Frame the problem

1 - Knowing the <span style="color: lime;">business objective</span> is important because it will determine how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.


2 - The next question to ask your boss is what the <span style="color: lime;">current situation</span> looks like (if any). The current situation will often give you a reference for performance, as well as insights on how to solve the problem


3 - What is the [[Pipelines]] :
	A sequence of data processing components.
	Pipelines are very common in machine learning systems, since there is a lot of data to manipulate and many data transformations to apply.


4 - With all this information, you are now ready to start designing your system. First, determine what kind of <span style="color:rgb(0, 255, 0)">training supervision</span> the model will need: is it a [[supervised]], [[unsupervised]], [[semi-supervised]], [[self-supervised]], or [[reinforcement learning]] task? 
And is it a [[classification]] task, a [[regression]] task, or something else? Should you use [[batch learning]] or [[online learning]] techniques?



![[Pasted image 20250506031209.png]]

#### 2 - Select a performance measure

- One of the most common method in regression models is RMSE (L2) this is work well with uniform distribution but is sensitive to outlier or large error values.

- another one is MAE (L1) this way is often used with data with a lot of outliers.


# 2 - Get the data
##### Function to automaticaly load the data

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
```

##### Function to split data using sklearn

```python
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
```
# 3 - Explore and visualize

# 4 - Prepare the data for ML algorithm

# 5 - Model Selection and training

# 6 - Fine-tuning

# 7 - Present your solution

# 8 - Launch, monitor and maintain
