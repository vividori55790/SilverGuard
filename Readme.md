요청하신 내용을 바탕으로, **Git 설치 후 초기 설정부터 GitHub 연동, 그리고 일상적인 협업 워크플로우(Pull, Commit, Push)**까지의 전 과정을 정리한 마크다운 문서를 작성해 드립니다.

아래 내용을 복사하여 `git-guide.md` 등의 파일로 저장해두고 참고하시면 좋습니다.

---

# 🚀 Git & GitHub 시작하기: 설치부터 첫 푸시까지

이 가이드는 Git을 처음 설치한 사용자가 GitHub와 연동하여 프로젝트를 관리하는 전체 과정을 다룹니다.

## 1. 초기 사용자 설정 (최초 1회)

Git을 설치한 후 가장 먼저 해야 할 일은 **사용자 이름**과 **이메일**을 등록하는 것입니다. 이 정보는 커밋(저장)할 때 "누가 수정했는지"를 기록하는 용도로 사용됩니다.

터미널(또는 Git Bash)을 열고 아래 명령어를 입력하세요.

```bash
# 본인의 GitHub 사용자 이름 입력
git config --global user.name "Your Name"

# 본인의 GitHub 가입 이메일 주소 입력
git config --global user.email "youremail@example.com"

```

* **확인 방법:** `git config --list` 를 입력하면 설정된 정보를 볼 수 있습니다.

---

## 2. 로컬 프로젝트 생성 및 Git 초기화

내 컴퓨터(로컬)에 프로젝트 폴더를 만들고 Git이 관리하도록 설정하는 단계입니다.

```bash
# 1. 프로젝트 폴더 생성 및 이동
mkdir my-project
cd my-project

# 2. Git 저장소 초기화 (이 폴더를 Git이 관리하기 시작함)
git init

```

> **Note:** `git init`을 실행하면 보이지 않는 `.git` 폴더가 생성되며, 이때부터 버전 관리가 가능해집니다.

---

## 3. GitHub 리포지토리(Remote) 생성 및 연결

이제 내 컴퓨터의 프로젝트를 업로드할 공간(원격 저장소)을 GitHub에 만듭니다.

1. [GitHub](https://github.com/)에 로그인합니다.
2. 우측 상단의 **+** 버튼을 누르고 **New repository**를 클릭합니다.
3. **Repository name**을 입력하고 (예: `my-project`), **Create repository** 버튼을 누릅니다.
* *Tip: 초기 설정 시 'Add a README file' 체크박스는 해제하는 것이 연결하기 더 쉽습니다.*



### 리모트(Remote) 연결하기

GitHub 저장소가 생성되면 주소(URL)가 나옵니다. 터미널로 돌아와 다음 명령어로 연결합니다.

```bash
# 기본 브랜치 이름을 'main'으로 변경 (과거엔 master였으나 최근엔 main을 표준으로 사용)
git branch -M main

# 원격 저장소(origin) 주소 등록
# [URL] 부분에 본인의 GitHub 저장소 주소를 넣으세요 (https://github.com/아이디/레포이름.git)
git remote add origin https://github.com/username/my-project.git

```

* **확인 방법:** `git remote -v` 를 입력하면 연결된 주소가 출력됩니다.

---

## 4. 파일 생성 및 첫 커밋 (Add & Commit)

파일을 만들고 버전을 저장하는 과정입니다. Git의 저장 과정은 **작업(Work) -> 스테이징(Add) -> 저장(Commit)**의 단계를 거칩니다.

```bash
# 1. 테스트용 파일 생성 (또는 코드 작성)
echo "# My First Project" > README.md

# 2. 스테이징 영역에 파일 추가 (Git에게 이 파일을 저장할 것이라고 알림)
# '.' 은 변경된 모든 파일을 의미합니다.
git add .

# 3. 커밋 (실제 저장소를 확정하고 메시지 남기기)
git commit -m "Initial commit: 프로젝트 시작 및 README 추가"

```

---

## 5. GitHub에 업로드 (Push)

내 컴퓨터(Local)에 저장된 커밋 내역을 GitHub(Remote)로 전송합니다.

```bash
# origin(원격지)의 main 브랜치로 푸시
# -u 옵션은 처음 한 번만 사용하면 되며, 이후에는 'git push'만 입력해도 됩니다.
git push -u origin main

```

> **인증 관련:** 명령어를 입력하면 GitHub 로그인 창이 뜨거나 토큰을 요구할 수 있습니다. 브라우저 창이 뜨면 로그인하여 승인(Authorize) 해주세요.

---

## 6. 협업 및 유지보수 워크플로우 (반복 과정)

초기 설정 이후, 개발을 진행하면서 반복하게 될 루틴입니다.

### A. 작업 전 업데이트 (Pull)

다른 사람이 코드를 수정했거나, 내가 다른 컴퓨터에서 작업했다면 최신 내용을 먼저 받아와야 합니다.

```bash
# 원격 저장소의 내용을 가져와서 내 로컬과 병합
git pull origin main

```

### B. 작업 후 저장 및 업로드

코드를 수정한 뒤에는 **Add -> Commit -> Push** 과정을 반복합니다.

```bash
# 1. 변경된 파일 확인 (선택 사항)
git status

# 2. 변경 파일 담기
git add .

# 3. 버전 기록 남기기
git commit -m "기능 추가: 로그인 페이지 구현"

# 4. GitHub로 보내기
git push

```

---

## 🎯 요약 치트시트

| 명령어 | 설명 |
| --- | --- |
| `git init` | 현재 폴더를 Git 저장소로 초기화 |
| `git remote add origin [URL]` | GitHub 원격 저장소 연결 |
| `git add .` | 변경된 모든 파일을 스테이징(준비) 영역에 추가 |
| `git commit -m "메시지"` | 변경 사항을 메시지와 함께 저장(버전 생성) |
| `git push` | 로컬의 커밋 내역을 GitHub에 업로드 |
| `git pull` | GitHub의 변경 사항을 로컬로 내려받기 |
| `git status` | 현재 파일들의 상태(변경됨, 스테이징됨 등) 확인 |
| `git log` | 커밋 기록 확인 |

---

### **다음 단계로 추천드리는 작업**

Git의 기초를 익히셨다면, **`.gitignore`** 파일을 설정하여 불필요한 파일(시스템 파일, 빌드 결과물 등)이 업로드되지 않도록 관리하는 방법을 알아보시겠습니까?
