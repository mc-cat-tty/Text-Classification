# Text-Classification
Natural Language Processing in Python


## Table of Contents
* [Project Status](#project-status)
* [Getting Started](#getting-started)
* [Authors](#authors)
* [License](#license)

## Project Status

### Build

![License](https://img.shields.io/badge/license-MIT-brightgreen) ![build](https://img.shields.io/badge/build-passed-brightgreen) ![Version](https://img.shields.io/badge/version-2.0.0-blue)


### Diagram

![Diagram](Diagram.jpg)


## Getting Started

### Requirements

`spacy`
`PyQt5`

### How to test if all work properly?

1. Install Requirements
2. Launch gui.py
  If you are in Bash-like environment with Python installed, you can run directly by typing:

  ```sh
  $ ./gui.py
  ```

  Otherwise, depending on your Python interpreter installation and your OS:

  ```sh
  $ python gui.py
  ```
  or
  ```sh
  $ py gui.py
  ```
3. Go to "Models->Add"
4. Select "happiness_vocabulary" (inside /models/ folder) and Load this model
5. Select "sadness_vocabulary" (inside /models/ folder) and Load this model
6. Go to "Classify->Start Classification", insert a text and press "Classify Text"

## Authors

 - Francesco Mecatti - I.T.I.S Enrico Fermi - Italy, Modena

## License
This project is licensed under the MIT license - see the [LICENSE.md](license.md) file for details