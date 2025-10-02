

# Pre-Requisites 🔧

The creation of dot files and svg images for the library is handled through the [GraphViz](https://graphviz.org/) suite. For everything to correctly work, this module has to be installed correctly. 
We reference both the [installation section on their docs](https://graphviz.org/download/), as well as the main ways to install it. 

### Linux

```bash
sudo apt install graphviz
```

### MacOS

```bash
# Installing Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install graphviz
```

### Windows

Installers are at the [Download page of GraphViz](https://graphviz.org/download/). 

# Installation 👷‍♂️

Install from the repository:

```bash
git clone https://github.com/HPCI-Lab/yProvML.git
cd yProvML

pip install -r requirements.txt
pip install .

# Or use apple extra if on a Mac
pip install .[apple]

# or install for specific arch
pip install .[nvidia] # or .[amd]
```

or simply:

```bash
pip install --no-cache-dir git+https://github.com/HPCI-Lab/yProvML
```

To install a specific branch of the library: 

```bash
git clone https://github.com/HPCI-Lab/yProvML.git
cd yProvML

git switch development # or any other branch

pip install -r requirements.txt
pip install .
```

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="setup.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>