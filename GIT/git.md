# 0103
### github
- 로컬 저장소에서 폴더 생성하여 생성한 폴더에 파일을 작성하여 commit/push 하면 원격 저장소에도 동일하게 폴더까지 생성된다.

`git clone git@github.com:taemt94/welcome.git .`  
- 해당 주소의 저장소를 현재 디렉토리로 가져온다. 여기서 ‘.’이 현재 디렉토리를 의미한다. '.'을 입력하지 않으면 레포지토리에 해당하는 폴더가 생성된다.  
- git clone은 클라이언트 상에 아무것도 없을 때(로컬 폴더가 빈 폴더일 때?) 서버의 프로젝트를 내려받는 명령어라고 한다.

`git diff`  
- 파일에서 수정한 내용을 보여준다.

`git add file3.txt`  
- File3.txt를 commit할 것임을 git에게 알려주는 명령어가 add.

`git add .`  
- 전체 파일 add.

`git commit –m “version5”`  
- 지금 수정된 내용을 “version5”라는 이름의 버전으로 만들라는 뜻.

`git log`  
- Commit 기록을 보여준다.

`git push`  
- git commit 까지만 하면 아직 로컬 저장소와 원격 저장소가 동기화되지 않은 상태이므로 로컬 저장소를 업로드해서 두 저장소의 파일 버전이 동일하게끔 동기화를 시켜줄 때 사용하는 명령어.

`git status`  
- 어느 파일이 수정되었는지를 알 수 있다.

`git remote –v`  
- 해당 디렉토리가 어느 레포지토리에 해당하는지 알 수 있다.

`git pull`  
- 다른 사람이 pull request를 통해서 코드를 업데이트했거나, github 상에서 commit을 했을 때 해당 업데이트를 로컬 저장소로 내려받는 명령어이다.  

`git init`  
- 처음 레포지토리를 생성할 때 해당 폴더로 들어가서 git init을 치면, .git 파일이 생성된다. git 파일에 모든 버전 정보가 저장된다.
- github 상에서 commit을 한 후 pull을 하지 않은채 로컬 저장소에서 commit을 하면 에러가 발생한다. 로컬 저장소에서만 파일을 관리하는 것이 좋을 듯 하다.
---
### git
- GIT은 하나의 version control system 이다. 변경사항을 관리하는 시스템. 추가적으로 백업, 복원, 협업 등을 할 수 있다.
- git bash는 리눅스 유닉스 계열의 명령어를 통해 윈도우를 제어할 수 있는 프로그램이다. 깃을 설치하면 같이 설치된다.
- 필요에 의해 작업이 분기되는 현상을 branch를 만든다고 말한다.
- 기존의 매우 무겁고 불편하고 느렸던 브랜치를 깃에서 매우 편리하게 쓸만하게 만들었다.
- 깃은 모든 것을 브랜치라는 개념으로 내부적으로도 다루고 있기 때문에 브랜치를 꼭 알고 있어야 한다.
---
# 0116
### github
`git remote add origin 'repository dir'`  
- origin이라는 이름으로 원격 저장소를 로컬 저장소에서 관리하게 해주는 명령어이다.  
- origin은 예시이고, 필요에 따라 원하는 이름을 설정하면 된다.  

`git remote remove origin`  
- origin이라는 이름으로 관리하는 원격 저장소를 제거한다.  
- 깃허브를 쓰다가 commit이 꼬여서 commit을 해도 push가 안되는 경우가 몇 번 있었다.
- 이럴 때 연동시켜놓은 저장소를 제거한 후 다시 remote add 해야 하기 때문에 remove 명령어를 사용한다.

# 0122
### GIT
`git reset HEAD [file name]`  
- git add 취소 명령어.
- file name을 입력하지 않으면 파일 전체를 취소한다.

`git reset --soft HEAD^`  
- git commit 취소 명령어.

# 0126
### GIT
```
## TIL Ignore File ##

# 확장자가 'a'인 모든 파일을 track하지 않게 할 때
*.a

# 위에서 확장자가 'a'인 모든 파일을 track하지 않도록 했지만, lib.a 파일은 track하도록 하고 싶을 때
!lib.a

# build/ 디렉토리 내의 모든 파일을 track하지 않게 할 때
build/

# doc 디렉토리의 모든 txt 파일은 track하지 않지만, doc 디렉토리 내의 다른 디렉토리의 txt 파일은 track하도록 할 때
doc/*.txt

# doc 디렉토리의 모든 pdf 파일을 track하지 않게 할 때  
doc/**/*.pdf
```
`git rm -r --cached .`  
`git add .`  
`git commit -m "Apply .gitignore"`  
- .gitignore 파일의 사용법은 위와 같다.
- .gitignore 파일을 만든 후 push할 때에는 위와 같은 명령어를 입력해주면 된다.  