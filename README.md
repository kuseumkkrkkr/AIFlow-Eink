<div align="center">

# AIFlow E-ink

### Handwriting intelligence for quiet, low-power learning

[![Hugging Face](https://img.shields.io/badge/Models-cwLeeDev-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827)](https://huggingface.co/cwLeeDev)
[![Profile](https://img.shields.io/badge/AIFlow-Profile-2563EB?style=for-the-badge&logo=github)](https://github.com/kuseumkkrkkr)

</div>

AIFlow E-ink는 수학 필기 인식 모델과 E-ink 학습 인터페이스를 연결하기 위한 공개 자료 저장소입니다.

이 저장소는 GitHub 홈에서 비어 보이지 않도록, Hugging Face에 공개된 AIFlow Math Ink 모델 카드, 연구 리포트, 매니페스트, Android 런타임 문서를 정리합니다. 실제 모델 체크포인트는 Hugging Face 원본 저장소에서 관리합니다.

<div align="center">

<img src="models/AIFlow-Math-Ink-0.5/assets/architecture.svg" alt="AIFlow Math Ink architecture" width="780">

</div>

## Model Hub

| Model | Role | Source |
| --- | --- | --- |
| AIFlow Math Ink 0.5 | 수학 필기 전처리, stroke grouping, geometry 기반 초기 공개 모델 | [Hugging Face](https://huggingface.co/cwLeeDev/AIFlow-Math-Ink-0.5) |
| AIFlow Math Ink 0.6 Intermediate | point/stroke sequence 기반 온디바이스 연구 checkpoint와 Android 런타임 자료 | [Hugging Face](https://huggingface.co/cwLeeDev/aiflow-math-ink-06-intermediate) |

## Repository Map

| Path | Content |
| --- | --- |
| [`models/AIFlow-Math-Ink-0.5`](models/AIFlow-Math-Ink-0.5) | 0.5 모델 카드, 라이선스, 방법론 문서, 아키텍처 |
| [`models/aiflow-math-ink-06-intermediate`](models/aiflow-math-ink-06-intermediate) | 0.6 모델 카드, Android 문서, 모델 인덱스, 매니페스트, 아키텍처 |
| [`research`](research) | 연구 리포트, federation audit, 핵심 실험 요약 JSON, schema/config |
| [`models/DOWNLOADS.md`](models/DOWNLOADS.md) | Hugging Face 체크포인트와 artifact 다운로드 링크 |

## System Direction

```text
E-ink handwriting input
  -> stroke / point sequence capture
  -> Math Ink model inference
  -> formula and behavior understanding
  -> AIFlow learning feedback loop
```

## Related Repositories

| Repository | Purpose |
| --- | --- |
| [AIFlow-Core](https://github.com/kuseumkkrkkr/AIFlow-Core) | AIFlow 문제 생성과 검증 엔진 |
| [AIFlow-Eink](https://github.com/kuseumkkrkkr/AIFlow-Eink) | E-ink 학습 인터페이스와 Math Ink 모델 자료 |

## License Notes

- AIFlow Math Ink 0.5 자료는 원본 Hugging Face 저장소의 Apache-2.0 라이선스와 고지를 함께 보존합니다.
- AIFlow Math Ink 0.6 Intermediate 자료는 원본 Hugging Face 모델 카드, `NOTICE.md`, `MANIFEST.json`, `MODEL_INDEX.json`을 함께 보존합니다.
- 대용량 모델 바이너리는 GitHub가 아닌 Hugging Face 원본 저장소에서 내려받는 방식을 권장합니다.
