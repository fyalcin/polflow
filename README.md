# PolFlow - Automated Modeling of Polarons

**PolFlow** is an automated workflow package for modeling polarons within density functional theory (DFT). It streamlines all computational steps required to identify, analyze, and interpret polaronic configurations in complex materials, leveraging machine learning to accelerate predictions and enable large-scale studies.

## Abstract

Polarons are widespread in functional materials and are key to device performance in several technological applications. However, their effective impact on material behavior remains elusive, as condensed matter studies struggle to capture their intricate interplay with atomic defects in the crystal.

In this work, we present an automated workflow for modeling polarons within density functional theory (DFT). Our approach enables a fully automatic identification of the most favorable polaronic configurations in the system. Machine learning techniques accelerate predictions, allowing for an efficient exploration of the defect-polaron configuration space. We apply this methodology to Nb-doped TiO₂ surfaces, providing new insights into the role of defects in surface reactivity.

Using CO adsorbates as a probe, we find that Nb doping has minimal impact on reactivity, whereas oxygen vacancies contribute significantly depending on their local arrangement via the stabilization of polarons on the surface atomic layer. Our package streamlines the modeling of charge trapping and polaron localization with high efficiency, enabling systematic, large-scale investigations of polaronic effects across complex material systems.

## Features

- Fully automated workflow for polaron modeling in DFT
- Automatic identification of favorable polaronic configurations
- Machine learning acceleration for efficient configuration space exploration
- Support for defect and adsorbate studies (e.g., Nb-doped TiO₂, CO adsorption)
- Modular design for extensibility to other material systems
- Streamlined handling of charge trapping and polaron localization

## Installation

### Requirements

- Python >= 3.10

### Using Rye or uv (Recommended)

If you use [Rye](https://rye-up.com/) or [uv](https://github.com/astral-sh/uv):

```bash
# With Rye
rye sync

# Or with uv
uv pip install -r requirements.txt
```

### Using pip

You can also install the dependencies with pip:

```bash
pip install -r requirements.txt
```

Or, if you want to install as a package (editable mode):

```bash
pip install -e .
```

## Usage

See the [examples/](examples/) directory for sample scripts and workflows.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note:** This package is under active development and will be published alongside the following paper:

> [Insert paper citation or link here]

