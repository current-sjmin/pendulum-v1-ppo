# Pendulum-PPO

이 프로젝트는 [Gymnasium](https://gymnasium.farama.org/)의 `Pendulum-v1` 환경을 기반으로  
PPO(Proximal Policy Optimization) 알고리즘을 구현한 강화학습 학습 예제입니다.

---

## 🛠️ 가상환경 설정

아래 명령어로 가상환경을 설정할 수 있습니다:

```bash
python -m venv venv
source venv/bin/activate
```

---

## 📦 의존성 설치

필요한 패키지는 다음 명령어로 설치합니다:

```bash
pip install -r requirements.txt
```

> ⚠️ `requirements.txt`에는 **CPU 버전의 PyTorch**가 포함되어 있습니다.  
> **GPU 사용이 가능한 환경이라면**, 아래 링크를 참고해 GPU 버전의 PyTorch를 설치하면  
> 학습 속도를 크게 향상시킬 수 있습니다:  
> 👉 https://pytorch.org/get-started/locally/

---

## 🚀 학습 실행

학습을 시작하려면 아래 명령어를 실행하세요:

```bash
python train.py
```

---

## 🎮 테스트 및 시뮬레이션 확인

학습된 모델의 시뮬레이션 결과를 렌더링하여 확인하려면 다음 명령어를 실행하세요:

```bash
python test.py
```

기본적으로 `test` 모듈은 가장 성능이 좋은 모델인 `best.pt`를 자동으로 로드합니다.  
다른 모델을 사용하고 싶다면, 실행 시 아래와 같이 인자를 전달할 수 있습니다:

```bash
python test.py --model saved_models/your_model.pt
```

또는

```bash
python test.py -m saved_models/your_model.pt
```

`-m` 또는 `--model` 옵션을 통해 원하는 모델을 자유롭게 지정할 수 있습니다.
