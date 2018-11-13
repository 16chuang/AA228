# AA228
AA228/CS238 Final Project

# Getting Started

## Pipenv

### Install Pipenv

Pipenv is a cool package that lets us create a virtual environment, so that everyone is running on the samve version of python with the same packages.

[Install and use pipenv](https://robots.thoughtbot.com/how-to-manage-your-python-projects-with-pipenv)

If pipenv says `bash: parse_git_branch: command not found`, the issue is that your .bashrc file in your home directory doesn't have the git branch parsing function.  Edit your `~/.bashrc` file to add the following:

```bash
# Git branch in prompt.
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
```

### Run Pipenv

Start your pipenv virtual environment with `pipenv shell`.  Run files with `pipenv run python filename.py`.

### Quit Pipenv

Exit the pipenv virtual environment with the `exit` command, or ctrl+d.