import numpy as np
import pandas as pd
df=pd.read_csv('/content/PlayTennis.csv')
df.shape
df.head()
def Entropy(data):
    d = data.iloc[:, -1]
    d = d.value_counts()
    s = 0
    for v in d.keys():
        p = d[v] / sum(d)
        s -= p * np.log2(p)
    return s
def IG(data, A): # The code within this function needs to be indented
    Es = Entropy(data)
    val = data[A].unique()
    s_c = data[A].value_counts()
    s_v = []

    for v in range(len(val)):
        ds = data[data[A] == val[v]]
        s = 0
        for res in data.iloc[:, -1].unique():
            try:
                pi = ds.iloc[:, -1].value_counts()[res] / len(ds)
                s -= pi * np.log2(pi)
            except:
                s = 0
        s_v.append(s)

    for i in range(len(val)):
        Es = Es - s_c[val[i]] * s_v[i] / sum(s_c)

    return Es
class Node(): # Fixed indentation for the class definition
  def __init__(self,name=None,attr=None):
    self.name=name
    self.attr=attr

  def call_(self):
    return self.name

def DTNode(data,features_used):
  node=Node()
  IGmax=0
  vbest=None
  val_list=[v for v in values(data)[:-1] if v not in features_used] # You might also have a NameError for values()
  if val_list !=[]:
    for v in val_list:
      if IG(data,v)>IGmax:
        IGmax=IG(data,v)
        v_best=v

    if v_best:
      features_used.append(v_best)
      node.name=v_best
      node.attr=values(data[v_best]) # You might also have a NameError for values()

      return (node)
    else:
      return (None)

  return (None)


      # Recursive function to build the decision tree (ID3 algorithm)
def DTClassifier(data, features_used):
    root = DTNode(data, features_used)
    DT_dict = {}

    # If root is found, continue building the tree
    if root is not None:
        item = []
        for attr in root.attr:
            dataN = data[data[root.name] == attr]
            if Entropy(dataN) == 0:  # If pure subset, it's a decision (leaf node)
                item.append((attr, dataN.iloc[:, -1].unique()[0]))
            else:
                dt = DTClassifier(dataN, features_used)
                item.append((attr, dt))

        DT_dict[root.name] = item

    return DT_dict
    # Function to print the ID3 decision tree with attributes and its decisions
def print_id3_tree(tree, depth=0):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{'|   ' * depth}{key}")  # Print the attribute name
            if isinstance(value, list):
                for v in value:
                    print(f"{'|   ' * (depth + 1)}{v[0]} ->", end=" ")
                    if isinstance(v[1], dict):
                        print()
                        print_id3_tree(v[1], depth + 2)  # Recursive call for further splits
                    else:
                        print(f"Decision: {v[1]}")  # Print the decision
    else:
        print(tree)

features_used = []

id3_decision_tree = DTClassifier(df, features_used)

# Print the ID3 decision tree with attributes and decisions
print("Final ID3 Decision Tree:")
print_id3_tree(id3_decision_tree)
