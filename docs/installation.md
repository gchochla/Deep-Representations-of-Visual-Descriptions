# Installation

1. Install mandatory requirements (Ubuntu 18.04):

    ```console
    $ sudo apt update
    $ sudo apt install git python3 python3-pip
    $ pip3 install virtualenv # no sudo
    ```

1. Clone repo:

    ```console
    $ git clone https://github.com/gchochla/CRNNs-for-Visual-Descriptions crnns4captions
    $ cd crnns4captions
    ```

1. Register commit hooks:

    ```console
    $ cp etc/pre-commit .git/hooks/pre-commit
    $ chmod +x .git/hooks/pre-commit
    ```

1. Create a new [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) for Python, `.venv`:

    ```console
    $ ~/.local/bin/virtualenv .venv
    ```

1. Activate virtual environment. Should be done every time a new terminal is instantiated:

    ```console
    $ source .venv/bin/activate
    ```

1. Install project dependencies (as a developer):

    ```console
    $ pip install -e .[dev]
    ```
