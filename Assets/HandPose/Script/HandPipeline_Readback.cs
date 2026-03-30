using UnityEngine;
using UnityEngine.Rendering;

namespace MediaPipe.HandPose {

//
// GPU to CPU readback implementation of the hand pipeline class
//

sealed partial class HandPipeline
{
    #region Read cache operations

    Vector4[] _readCache;
    bool _readFlag;

    int TotalKeyPointCount => KeyPointCount * 2 * _maxHands;
    int ReadbackBytes => TotalKeyPointCount * sizeof(float) * 4;

    void InitReadCache()
      => _readCache = new Vector4[TotalKeyPointCount];

    Vector4[] ReadCache
      => (_readFlag || UseAsyncReadback) ? _readCache : UpdateReadCache();

    Vector4[] UpdateReadCache()
    {
        _buffer.filter.GetData(_readCache, 0, 0, TotalKeyPointCount);
        _readFlag = true;
        return _readCache;
    }

    void InvalidateReadCache()
    {
        if (UseAsyncReadback)
            AsyncGPUReadback.Request
              (_buffer.filter, ReadbackBytes, 0, ReadbackCompleteAction);
        else
            _readFlag = false;
    }

    #endregion

    #region GPU async operation callback

    System.Action<AsyncGPUReadbackRequest> ReadbackCompleteAction
      => OnReadbackComplete;

    void OnReadbackComplete(AsyncGPUReadbackRequest req)
      => req.GetData<Vector4>().CopyTo(_readCache);

    #endregion
}

} // namespace MediaPipe.HandPose
