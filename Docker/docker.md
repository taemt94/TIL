#### 2022/02/17
# Docker 명령어 정리
### docker pull
- Pull an image or a repository from a registry
    ```
    $ docker pull [OPTIONS] NAME[:TAG|@DIGEST]
    ```
### docker images
- List images
    ```
    $ docker images [OPTIONS] [REPOSITORY[:TAG]]
    ```

### docker run
- Run a command in a new container
    ```
    $ docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
    ## --name option : Container 이름 설정 가능
    ## -v option : Host 디렉토리와 Container 내 디렉토리 연결 가능 -> Host 디렉토리 내에서 코드 작업 시 Container에도 반영됨. -> Host에서 vscode 사용해서 Container 작업 가능.

    ### Example
    $ docker run -v ~/Desktop/test:/usr/directory/of/container --name container_name IMAGE
    ###
    ```

### docker stop
- Stop one or more running containers
    ```
    $ docker stop [OPTIONS] CONTAINER [CONTAINER...]
    ```

### docker ps
- List containers
    ```
    $ docker ps [OPTIONS] ## -a option : Show all containers (default shows just running)
    ```

### docker start
- Start one or more stopped containers
    ```
    $ docker start [OPTIONS] CONTAINER [CONTAINER...]
    ```

### docker logs
- Fetch the logs of a container
    ```
    $ docker logs [OPTIONS] CONTAINER   ## -f option : Follow log output
    ``` 

### docker rm
- Remove one or more containers
    ```
    $ docker rm [OPTIONS] CONTAINER [CONTAINER...]  ## --force option : Force the removal of a running container (uses SIGKILL)
    ```

### docker rmi
- Remove one or more images
    ```
    $ docker rmi [OPTIONS] IMAGE [IMAGE...]
    ```

### docker exec
- Run a command in a running container
    ```
    $ docker exec [OPTIONS] CONTAINER COMMAND [ARG...]

    ### Example
    $ docker exec ws3 pwd
    $ docker exec ws3 ls
    ### 

    $ docker exec -it CONTAINER /bin/sh 또는 /bin/bash  ## -i & -t option : Container 상에서 지속적으로 명령어 입력할 수 있음.

    $ exit  ## Container 밖으로 나가는 명령어.
    ```
