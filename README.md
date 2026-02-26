# HifiSamplerSharp

HifiSamplerSharp is a C# rewrite of [openhachimi/hifisampler](https://github.com/openhachimi/hifisampler).

## Components

- `HifiSampler.Core`: Core resampling pipeline and ONNX inference logic.
- `HifiSampler.Server`: HTTP service that receives requests and runs processing.
- `HifiSampler.Client`: Command-line client for sending render requests.

## Installation

Using release artifacts is recommended, But you can also publish locally.

Then you need to prepare the ONNX model and place it in the expected location:

- Models/Vocoder/model.onnx
- Models/Hnsep/model.onnx

But you can also specify custom paths in appsettings.json:

```json
{
  "VocoderConfig": {
    "ModelType": "onnx",
    "ModelPath": "path/to/vocoder/model.onnx",
    ...
  }
}
```

```json
{
  "HnSepConfig": {
    "ModelType": "onnx",
    "ModelPath": "path/to/hnsep/model.onnx",
    ...
  }
}
```


