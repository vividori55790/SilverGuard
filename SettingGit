Git & GitHub 입문 가이드이 문서는 Git을 처음 설치하고 GitHub와 연동하여 코드를 관리하는 전 과정을 안내합니다.1. Git 설치하기Windows 사용자git-scm.com에 접속하여 Download for Windows를 클릭합니다.다운로드된 설치 파일을 실행합니다.설치 과정에서 나오는 옵션들은 대부분 Next를 눌러 기본값(Default)으로 진행해도 무방합니다.macOS 사용자터미널(Terminal)을 열고 아래 명령어를 입력합니다. (Homebrew가 설치되어 있다고 가정)brew install git
또는 Windows와 마찬가지로 git-scm.com에서 설치 파일을 다운로드할 수 있습니다.2. 초기 사용자 설정 (최초 1회 필수)Git을 설치한 후, 커밋(저장) 기록에 남을 사용자의 이름과 이메일을 설정해야 합니다. 터미널(또는 Git Bash)을 열고 입력하세요.# 본인의 GitHub 사용자 이름 입력
git config --global user.name "Your Name"

# 본인의 GitHub 가입 이메일 입력
git config --global user.email "your_email@example.com"
설정 확인하기:git config --list
3. 로컬 저장소 만들기 (git init)내 컴퓨터(로컬)에 있는 프로젝트 폴더를 Git이 관리하도록 설정합니다.터미널(VS Code 터미널 등)에서 프로젝트 폴더로 이동합니다.아래 명령어를 입력합니다.git init
성공 시: Initialized empty Git repository in... 메시지가 뜹니다.4. GitHub 저장소(Remote) 연결하기4-1. GitHub에서 저장소 생성GitHub에 로그인합니다.우측 상단의 + 버튼을 누르고 New repository를 선택합니다.Repository name을 입력하고 Create repository 버튼을 클릭합니다.4-2. 원격 저장소 주소 연결 (Remote Add)방금 만든 GitHub 저장소 페이지에 보이는 주소(https://github.com/.../repo.git)를 복사한 뒤, 터미널에 입력합니다.# 원격 저장소의 별명을 'origin'으로 짓고 주소를 연결
git remote add origin [https://github.com/사용자명/저장소명.git](https://github.com/사용자명/저장소명.git)
연결 확인하기:git remote -v
5. 기본 워크플로우 (Add, Commit, Push)코드를 수정하고 GitHub에 올리는 3단계 과정입니다.1단계: 파일 담기 (Add)수정한 파일을 스테이징 영역(Staging Area)에 올립니다.# 특정 파일만 올리기
git add 파일명.txt

# 변경된 모든 파일 올리기 (가장 많이 사용)
git add .
2단계: 저장하기 (Commit)스테이징 된 파일들을 확정하여 로컬 저장소에 기록을 남깁니다.git commit -m "작업한 내용에 대한 간략한 메시지"
3단계: GitHub에 업로드 (Push)로컬의 커밋 내역을 GitHub 원격 저장소로 보냅니다.# origin(원격)의 main 브랜치로 푸시
git push origin main
참고: 예전에는 기본 브랜치명이 master였으나 최근에는 main을 주로 사용합니다. 오류가 난다면 git push origin master를 시도해보거나 현재 브랜치 이름을 확인하세요 (git branch).6. 협업 및 동기화 (Pull)GitHub(원격 저장소)에 있는 내용이 내 컴퓨터(로컬)보다 최신일 때(예: 다른 사람이 수정했거나, 내가 다른 컴퓨터에서 작업했을 때), 내용을 가져옵니다.# 원격 저장소(origin)의 내용을 가져와서 현재 브랜치에 합침
git pull origin main
7. 전체 흐름 요약표단계명령어설명준비git init현재 폴더를 Git 저장소로 초기화연결git remote add origin [URL]GitHub 주소 연결선택git add .변경된 모든 파일을 저장 대기저장git commit -m "메시지"로컬에 버전 저장업로드git push origin mainGitHub로 전송내려받기git pull origin mainGitHub 내용을 로컬로 가져오기상태확인git status현재 파일들의 상태 확인 (수시로 사용 추천)💡 자주 묻는 질문 (FAQ)Q. Push 할 때 비밀번호를 물어보는데 로그인이 안 돼요.GitHub는 2021년부터 비밀번호 대신 토큰(Personal Access Token) 인증을 사용합니다.하지만 가장 쉬운 방법은 Git Credential Manager를 사용하는 것입니다. 최신 버전의 Git을 설치했다면, 처음 Push 할 때 뜨는 브라우저 창에서 GitHub 로그인을 승인하면 자동으로 인증됩니다.Q. git push를 했는데 rejected 오류가 나요.원격 저장소에 내 컴퓨터에는 없는 새로운 커밋이 있기 때문입니다. 먼저 git pull origin main을 해서 변경 사항을 내 컴퓨터로 가져온(병합한) 후에 다시 push 하세요.
