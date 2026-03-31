<!-- Author: SangJeong Kim -->
<!-- Last Modified: 2026-03-30 -->

<div align="center">
  <h1> dataset loader </h1>

[sangjung0](https://github.com/sangjung0)

  <br>

<a href="https://github.com/sangjung0/dataset_loader/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sangjung0/sjaipy" />
</a>

  <br>
  <br>
</div>

이 프로젝트는 “여러분께 제공하려고 만든 제품”이라기보다는, **제가 개인적으로 쓰려고 만들다가** 혹시 필요할 사람이 있을까 해서 공개해둔 유틸리티에 가깝습니다.
따라서 API/동작은 제 사용 흐름에 맞춰져 있고, 문서/호환성/안정성 보장은 제한적일 수 있습니다.
프로덕션 용도라면 가능하면 더 성숙한 대안(예: Hugging Face `datasets` 등)을 먼저 검토하고, 이 레포는 필요할 때만 “가볍게 가져다 쓰는” 정도로 봐주시면 좋겠습니다.

여러 음성/이미지 데이터셋을 **비슷한 사용 흐름으로 다룰 수 있게** 만들기 위한 라이브러리입니다.

데이터셋마다

- 다운로드 방식이 다르고
- 컬럼/메타데이터 이름이 다르고
- 샘플에서 오디오/이미지를 꺼내는 방식이 다르다 보니

프로젝트마다 로딩 코드가 쉽게 흩어지고 유지보수가 어려워집니다.
`dataset_loader`는 이런 부분을 정리해서, “데이터셋이 바뀌어도 코드를 크게 안 바꾸고” 로딩/순회/샘플 접근을 할 수 있게 하는 게 목적입니다.

## 제공하는 것

- `Sample`: 모든 데이터셋이 공통으로 반환하는 샘플 형식입니다. (아래 참고)
- `DatasetProtocol`, `SampleProtocol`: 타입힌트/에디터 자동완성에서 사용할 수 있는 동작 규약입니다.
- Loader (데이터셋별 로더): 데이터셋을 내려받거나(가능한 경우) 데이터셋 객체를 만들어줍니다.
    - `ESICv1`, `KSPonSpeech`, `LibriSpeech`, `Tedlium`, `ZerothKorean`
- Wrapper (도메인 편의 기능): 샘플 접근을 편하게 만들고, 스레드 프리패치를 제공합니다.
    - `ASRDataset` + `ASRSample` (음성/전사)
    - `IRDataset` + `IRSample` (이미지/라벨)

## Sample

기본 샘플은 `Sample(id, data)` 형태입니다.

- `id`: 샘플 식별자(문자열)
- `data`: 샘플의 실제 내용을 담는 `dict`

데이터셋에 따라 `data` 내부 키는 달라질 수 있습니다. 예를 들어 음성 데이터셋은 `ref`(전사)나 오디오 로딩 함수 같은 값이 들어갈 수 있고, 이미지 데이터셋은 원본 이미지/라벨 정보가 들어갈 수 있습니다.

Wrapper를 쓰면(`ASRDataset`, `IRDataset`) 이 `data`를 직접 만지지 않고도 `sample.audio`, `sample.ref`, `sample.raw`, `sample.label` 같은 방식으로 접근할 수 있게 됩니다.

## 설치

이 프로젝트는 `sjpy`에 의존합니다. 이 워크스페이스에서는 `modules/sjpy`가 git submodule이므로 먼저 초기화하세요.

```bash
git submodule update --init --recursive
```

그 다음에는 사용하는 환경에 맞는 방식으로 설치하세요(pip/uv/poetry 등).
필요한 시스템 패키지(예: `ffmpeg` 등)와 전체 설치 흐름은 루트 [Dockerfile](Dockerfile)을 참고하는 게 가장 빠릅니다.

참고: 데이터셋/환경에 따라 오디오 로딩에 `ffmpeg` 같은 시스템 도구가 필요할 수 있습니다.

## 빠른 시작

### LibriSpeech (다운로드 + 준비 + ASR wrapper)

```python
from dataset_loader import LibriSpeech, ASRDataset

loader = LibriSpeech()

# raw 파일 다운로드
loader.download(name="train-clean-100")

# 필요 시 인덱스/메타데이터 준비(캐시 내부에 생성)
loader.prepare(name="train-clean-100")

ds = loader.train_clean_100(sr=16000)
asr = ASRDataset(dataset=ds)

sample = asr[0]
wav = sample.audio
text = sample.ref
```

스레드 프리패치(백그라운드에서 미리 로딩/계산):

```python
for sample in asr.thread_iter(num_workers=4, prefetch=16):
    wav = sample.audio
```

## Tedlium 안내

`Tedlium`은 현재 원본 데이터가 예전처럼 공개되어 있지 않아, **자동 다운로드 기능을 제공하지 않습니다**.
이미 데이터를 따로 보유/준비한 경우에만 로더를 통해 불러오는 형태로 사용합니다.

## Wrapper

### ASRDataset / ASRSample

- `ASRDataset(dataset=...)`로 감싸면 샘플에서 `sample.audio`, `sample.ref`를 바로 쓸 수 있습니다.
- `thread_iter(num_workers=..., prefetch=...)`로 프리패치를 켤 수 있습니다.

### IRDataset / IRSample

- `IRDataset(dataset=...)`로 감싸면 샘플에서 `sample.raw`, `sample.label`을 바로 쓸 수 있습니다.
- `thread_iter(num_workers=..., prefetch=...)`를 지원합니다.

## 라이선스

MIT (루트의 `LICENSE` 참고)

참고: `modules/sjpy`는 submodule이며 별도 레포/별도 라이선스입니다. 함께 포함해 배포하는 경우 `sjpy`의 라이선스도 준수해야 합니다.
