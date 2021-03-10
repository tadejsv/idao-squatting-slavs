# 🥇 IDAO Squatting Slavs 🏆

- [🥇 IDAO Squatting Slavs 🏆](#-idao-squatting-slavs-)
  - [Devcontainer](#devcontainer)
    - [Quickstart: local](#quickstart-local)
    - [Quickstart: remote](#quickstart-remote)


## Devcontainer

### Quickstart: local

This assumes that all your files are on a local machine, where you also want to run the container. If you want to run the container on a remote machine, check out the [remote](#quickstart-remote) section.

All you need to do is to open the repository in VSCode, press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> and type/select "Reopen in Container".

### Quickstart: remote

This instructions are for the following scenario: your files and credentials are on a remote **host** machine (such as an AWS server, desktop workstation), and the only use of your **local** machine is to connect to the host. It is required that you have **the same username** on both local and host machine.

> The same username is required, because it will be used in the name of docker compose project, as well as the docker image. This enables multiple users on the same machine to use development containers without interfering with each other, as they will all have a separate compose project/docker image.

First, you need to set the `docker.host` setting in VSCode on your local machine to point at your host machine - see [here](https://code.visualstudio.com/docs/remote/containers-advanced#_a-basic-remote-example) for instructions. Next, you need to execute the following on both the local and host machine from the root of your repository (this you have to do just once in the lifecycle of the project)

```
echo "COMPOSE_PROJECT_NAME=${PWD##*/}_${USER}" >> .env 
```

Next, execute this from the root of your repository on the host machine

``` 
docker-compose -f .devcontainer/docker-compose.yml up -d
```

You can close the remote terminal after that. Finally, open the repository in VSCode on your local machine, press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> and type/select "Reopen in Container".

This way, everything will work as expected - even the port 8888 of the remote container will be mapped to port 8888 in your local machine.

