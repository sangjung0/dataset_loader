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

This project is less of a “product built and polished for others” and more of a utility I originally made for my own use, then decided to open up in case it might be useful to someone else.  
So the API and behavior are shaped around my own workflow, and guarantees around documentation, compatibility, and stability may be limited.  
If you are considering it for production use, it is probably better to evaluate more mature alternatives first (for example, Hugging Face `datasets`), and treat this repository as something you can “lightly borrow and use when needed.”

This library is intended to make it easier to handle multiple speech and image datasets through a **similar usage flow**.

Across datasets,

- the download process differs,
- the column and metadata names differ,
- and the way audio/images are extracted from each sample differs,

so dataset loading code tends to become scattered across projects and harder to maintain.  
`dataset_loader` is meant to organize those differences so that even if the dataset changes, you can still keep a largely similar code path for loading, iterating, and accessing samples.

## What it provides

- `Sample`: the common sample format returned by all datasets. (See below.)
- `DatasetProtocol`, `SampleProtocol`: behavioral contracts for type hints and editor autocompletion.
- Loaders (dataset-specific loaders): download datasets when possible and/or create dataset objects.
    - `ESICv1`, `KSPonSpeech`, `LibriSpeech`, `Tedlium`, `ZerothKorean`
- Wrappers (domain-specific convenience features): make sample access more ergonomic and provide thread-based prefetching.
    - `ASRDataset` + `ASRSample` (speech/transcription)
    - `IRDataset` + `IRSample` (image/label)

## Sample

The base sample format is `Sample(id, data)`.

- `id`: a string identifier for the sample
- `data`: a `dict` containing the actual sample contents

The keys inside `data` may differ depending on the dataset.  
For example, a speech dataset may contain values such as `ref` (transcription) or an audio loading function, while an image dataset may contain the raw image and label information.

If you use a wrapper (`ASRDataset`, `IRDataset`), you can access these more conveniently without directly touching `data`, using properties like `sample.audio`, `sample.ref`, `sample.raw`, or `sample.label`.

## Installation

This project depends on `sjpy`. In this workspace, `modules/sjpy` is included as a git submodule, so initialize it first:

```bash
git submodule update --init --recursive
```

After that, install the project using whichever environment/tooling you prefer (`pip`, `uv`, `poetry`, etc.).  
For required system packages (for example, `ffmpeg`) and the overall installation flow, the root [Dockerfile](Dockerfile) is the quickest reference.

Note: depending on the dataset and environment, audio loading may require system tools such as `ffmpeg`.

## Quick Start

### LibriSpeech (download + prepare + ASR wrapper)

```python
from dataset_loader import LibriSpeech, ASRDataset

loader = LibriSpeech()

# download raw files
loader.download(name="train-clean-100")

# prepare indices/metadata if needed (generated inside the cache)
loader.prepare(name="train-clean-100")

ds = loader.train_clean_100(sr=16000)
asr = ASRDataset(dataset=ds)

sample = asr[0]
wav = sample.audio
text = sample.ref
```

Thread prefetching (preload/compute in the background):

```python
for sample in asr.thread_iter(num_workers=4, prefetch=16):
    wav = sample.audio
```

## Tedlium note

For `Tedlium`, the original data is no longer publicly available in the same way as before, so **automatic download is not provided**.  
It is intended to be used only when you have already obtained and prepared the dataset yourself.

## Wrapper

### ASRDataset / ASRSample

- Wrap a dataset with `ASRDataset(dataset=...)` to access `sample.audio` and `sample.ref` directly.
- Supports prefetching via `thread_iter(num_workers=..., prefetch=...)`.

### IRDataset / IRSample

- Wrap a dataset with `IRDataset(dataset=...)` to access `sample.raw` and `sample.label` directly.
- Supports `thread_iter(num_workers=..., prefetch=...)`.

## License

MIT (see the root `LICENSE`)

Note: `modules/sjpy` is a submodule with its own separate repository and license.  
If you distribute it together with this project, you must also comply with the license of `sjpy`.
