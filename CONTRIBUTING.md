# Contributing to TorchQuantumDistributed

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to the TorchQuantumDistributed package, which is hosted in the [IonQ Organization](https://github.com/ionq) on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [IonQ](#ionq)
  * [TorchQuantumDistributed](#torchquantumdistributed)
    * [Design Decisions](#design-decisions)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs and Suggesting Improvements](#reporting-bugs-and-suggesting-improvements)
  * [Pull Requests](#pull-requests)

[Styleguides](#styleguides)
  * [Git Commit Messages](#git-commit-messages)
  * [Languages](#languages)


## Code of Conduct

The general rule is: don't be rude and try to be constructive. The flip side is: don't take things personally and be open to suggestions.

## What should I know before I get started?

### IonQ

[IonQ](https://ionq.co) is a quantum hardware and solutions company. Our mission is to develop the world’s best quantum technology to solve the world’s most complex problems.

### TorchQuantumDistributed

To achieve our mission, we build many sophisticated tools to help us model and understand how our systems behave. We believe it is beneficial to our mission to contribute to the open-source software ecosystem to enable more people to appreciate how our computers can solve problems that _they_ care about.

TorchQuantumDistributed bridges the worlds of quantum computing and differentiable machine learning in a way that is scalable and hardware agnostic (to the extent that PyTorch can support).

#### Design Decisions

Our broad goals for this project are for the code to be:
- both Object Oriented and Functional, whenever beneficial
- Extensible
- Modular
- Parsimonious

When we make a significant decision in how we maintain the project and what we can or cannot support, we may document it in the `README.md`. If you have a question around how we do things, check to see if it is already addressed there.

## How Can I Contribute?

### Reporting Bugs and Suggesting Improvements

This section guides you through submitting a bug report or suggested enhancement. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting An Issue

* **Perform a [cursory search](https://github.com/ionq/torchquantum-dist/issues)** to see if the problem or idea has already been reported. If it has **and the issue is still open**, add a comment or like to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Issue?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue in the repository, explaining the problem and any additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact set up and steps that reproduce the problem or implement the idea** in as many details as possible. For example, start by explaining your environment, e.g. your OS, python version, package manager, other installed packages, which command exactly you use in the terminal or python shell, etc.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**

* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.

### Pull Requests

Please follow these steps to have your contribution considered by the maintainers:

1. Follow the [styleguides](#styleguides)
2. While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Feel free to start the commit message with an applicable emoji:
    * :art: `:art:` when improving the format/structure of the code
    * :racehorse: `:racehorse:` when improving performance
    * :non-potable_water: `:non-potable_water:` when plugging memory leaks
    * :memo: `:memo:` when writing docs
    * :checkered_flag: `:checkered_flag:` when fixing something on Windows
    * :bug: `:bug:` when fixing a bug
    * :fire: `:fire:` when removing code or files
    * :white_check_mark: `:white_check_mark:` when adding tests
    * :arrow_up:/:arrow_down: `:arrow_up:`/`:arrow_down:` when upgrading/downgrading dependencies

### Languages

* Use [Markdown](https://daringfireball.net/projects/markdown).
* Roughly follow [Python PEP 8](https://peps.python.org/pep-0008/#naming-conventions).
