# Player appearance model

Model: OSNet x0.25 trained on MSMT17, 512-dimensional person Re-ID features.
Architecture and preprocessing: https://github.com/KaiyangZhou/deep-person-reid
ONNX export publisher: https://huggingface.co/anriha/osnet_x0_25_msmt17
Pinned revision: `1e22b925c70caa5591e9fab3f5540af8484bcad8`
File: `osnet-x025.onnx`
SHA-256: `e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277`

The export declares MIT licensing; the upstream MIT notice is preserved in
OSNET-LICENSE.txt. This is a third-party ONNX export, not a soccer-trained model.
Its empty model card supplies no soccer accuracy claims. Evaluate on your own
annotated footage before treating matching scores as reliable identities.

Graph interface inspected from the bundled artifact: input `input` float32
`[16,3,256,128]`, output `output` float32 `[16,512]`. RGB crops are normalized
using ImageNet mean `[0.485,0.456,0.406]` and standard deviation
`[0.229,0.224,0.225]`, following upstream Torchreid FeatureExtractor. The browser
pads batches to 16, discards padded outputs and L2-normalizes real descriptors.
The model and all inference run locally. No footage or crops are uploaded to
the model publisher.
