# Rage against the State Machine (RATS)

![build](https://github.com/LEB-EPFL/rats/actions/workflows/build.yml/badge.svg)

High performance probabilistic state machine simulator

## Development

### Setup the development environment

#### Linux and macOS

1. Install [Rust](https://www.rust-lang.org/learn/get-started): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Install [pyenv](https://github.com/pyenv/pyenv): `curl https://pyenv.run | bash`
3. Install Python interpreter(s) listed in the file `.python-version`:

```console
while read line; do
  pyenv install "$line"
done < .python-version
```
4. Create a virtual environment: `python3 -m venv .venv`
5. Activate the venv: `source .venv/bin/activate`
6. Install the dependencies: `pip install -e .[develop]`

### Testing and linting

#### Rust

Run tests:

```console
cargo test
```

Run linters:

 ``` console
 cargo fmt -v --check
 cargo clippy
 ```

 #### Python

 Run all tests and linters:

 ```console
 tox
 ```

 Run specific linters:

```console
# Black, isort, mypy, pylint
tox -e black
# etc. ...
```

### Formatting

#### Rust

```console
cargo fmt
```

#### Python

```console
tox -e format
```
