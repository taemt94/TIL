#### 2022/08/01
# Pytorch 분산 학습 (Distributed training) 코드 디버깅을 위한 VSCode `launch.json` 설정

- 아래에 보이는 `json` 파일은 VSCode로 디버깅 시 기본적으로 생성되는 `launch.json` 파일이다.
``` json
// launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": ["{...}/config.py",
                     "--gpu-id", "1",
                     "..."],
            "justMyCode": true
        }
    ]
}
```

- 그러나 `Pytorch`로 분산 학습 시에는 `torch` 패키지에 있는 `torch/distributed/launch.py` 파일을 실행시켜야 하므로 `launch.json` 파일을 아래와 같이 수정해주어야 한다.
- 중요한 것은, `launch.json`의 실행 프로그램을 `torch/distributed/launch.py`으로 설정하고, 실제로 디버깅을 해야되는 파일(`./tools/train.py`)을 입력 인자(`"args"`)로 넣어주어야 된다는 점이다.
- 아래의 예시에서 `./tools/train.py` 위쪽에 있는 인자은 `torch/distributed/launch.py`에 입력될 인자이고, 아래쪽에 있는 인자는 `./tools/train.py`에 입력될 인자이다.
``` json
// launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "/{root_directory}/anaconda3/envs/{env_name}/lib/python3.7/site-packages/torch/distributed/launch.py",
            "console": "integratedTerminal",
            "args": ["--nnodes", "1",
                     "--node_rank", "0",
                     "--master_addr", "127.0.0.1",
                     "--nproc_per_node", "1",
                     "--master_port", "29500",
                     // 분산 학습 코드 경로로 인자로 입력
                     "./tools/train.py",
                     "{...}/config.py", 
                     "--seed", "0",
                     "--launcher", "pytorch",
                     "..."],
        }     
    ]
}
```