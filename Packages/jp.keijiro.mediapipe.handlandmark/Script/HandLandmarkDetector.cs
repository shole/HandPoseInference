using Unity.InferenceEngine;
using UnityEngine;
using Klak.NNUtils;
using Klak.NNUtils.Extensions;

namespace MediaPipe.HandLandmark {

//
// Implementation of the hand landmark detector class
//
public sealed partial class HandLandmarkDetector : System.IDisposable
{
    #region Private objects

    ResourceSet _resources;
    Worker _worker;
    ImagePreprocess _preprocess;
    GraphicsBuffer _output;
    BufferReader<Vector4> _readCache;

    void AllocateObjects(ResourceSet resources)
    {
        _resources = resources;

        // NN model
        var model = ModelLoader.Load(_resources.model);

        // GPU worker
        _worker = new Worker(model, BackendType.GPUCompute);

        // Preprocess
        _preprocess = new ImagePreprocess(ImageSize, ImageSize, nchwFix: true);

        // Output buffer
        _output = BufferUtil.NewStructured<Vector4>(VertexCount + 1);

        // Landmark data read cache
        _readCache = new BufferReader<Vector4>(_output, VertexCount + 1);
    }

    void DeallocateObjects()
    {
        _worker?.Dispose();
        _worker = null;

        _preprocess?.Dispose();
        _preprocess = null;

        _output?.Dispose();
        _output = null;
    }

    #endregion

    #region Neural network inference function

    void RunModel(Texture source)
    {
        _preprocess.Dispatch(source, _resources.preprocess);
        RunModel();
    }

    void RunModel()
    {
        // NN worker execution
        _worker.Schedule(_preprocess.Tensor);

        // Postprocessing
        var post = _resources.postprocess;
        post.SetBuffer(0, "_Landmark", ((ComputeTensorData)(_worker.PeekOutput("Identity") as Tensor<float>).dataOnBackend).buffer);
        post.SetBuffer(0, "_Score", ((ComputeTensorData)(_worker.PeekOutput("Identity_1") as Tensor<float>).dataOnBackend).buffer);
        post.SetBuffer(0, "_Handedness", ((ComputeTensorData)(_worker.PeekOutput("Identity_2") as Tensor<float>).dataOnBackend).buffer);
        post.SetBuffer(0, "_Output", _output);
        post.Dispatch(0, 1, 1, 1);

        // Cache data invalidation
        _readCache.InvalidateCache();
    }

    #endregion
}

} // namespace MediaPipe.HandLandmark
